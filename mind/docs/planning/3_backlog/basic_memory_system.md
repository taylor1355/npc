# Memory System with Scoring

**Status:** ✅ Partially Implemented (Basic system complete, scoring not yet added)

## Problem Statement

NPCs need to remember their experiences to maintain continuity and make informed decisions. Without memory, each decision is made in isolation, creating inconsistent and unbelievable behavior.

## Design Goals

- Store and retrieve memories based on relevance
- Assign importance to memories for later retrieval weighting
- Simple, reliable foundation for more sophisticated memory systems
- Clear API that won't need major changes as system evolves

## Technical Approach

### Core Memory Operations

1. **Storage**: Memories as text with metadata (timestamp, importance, emotional valence)
2. **Importance Scoring**: LLM assigns 1-10 significance when memory is created
3. **Retrieval**: Query memories based on semantic similarity
4. **Scoring Function**: Combine relevance, recency, and importance
   - Start simple (just relevance)
   - Add recency decay
   - Weight by importance
   - Later: emotional intensity affects retrieval probability

### Memory Types to Support

- **Episodic**: Specific events and experiences
- **Semantic**: Facts and knowledge about the world
- **Social**: Information about other agents
- **Procedural**: How to do things (though this might become behavioral chunks)

### Implementation Evolution

1. **MVP**: Vector store with semantic search
2. **V2**: Add recency and importance to scoring
3. **V3**: Emotional intensity affects retrieval
4. **Future**: Graph relationships between memories

## Benefits

- **Obvious Foundation**: Every other cognitive feature needs memory
- **Concrete**: Can immediately test with real memories and queries
- **Constrains Design**: Decisions here affect retrieval patterns, scoring, and more

## Dependencies

- Embedding model (already using HuggingFace)
- Vector store implementation (llama-index)
- Basic LLM for importance scoring

## Current Implementation

✅ **Completed:**
- Vector store via ChromaDB with semantic search
- Memory query generation from observations
- Top-k retrieval per query
- Basic Memory model with text content

⚠️ **Missing:**
- Importance scoring during formation
- Recency decay in retrieval
- Emotional intensity weighting
- Memory metadata (timestamp, location, ID)
- Memory deduplication

## Next Steps

See [roadmap.md](../roadmap.md) Phase 1 for:
1. Memory metadata enhancement (1.1)
2. Memory importance scoring (1.2)
3. Memory deduplication (1.4)

## Priority Rationale

**Obviousness**: Very high - Memory is clearly required for any cognitive continuity. The need is unambiguous even if implementation has choices.

**Development Velocity**: Massive positive long-term
- Tech tree effect: Every other cognitive feature depends on this - it's THE bottleneck
- Enables: working memory, planning, reflection, social memory, everything
- ✅ Basic system now complete and unblocking development

**Concreteness**: Moderate - Players won't directly see memory, but will see more consistent NPC behavior

This is the foundational feature that everything else builds upon. **Basic implementation is now complete; enhancements remain.**