# Cognitive Architecture Development Roadmap

**Last Updated:** October 5, 2025 (Post-LangGraph Migration)

## Overview

This roadmap outlines the phased evolution of the NPC cognitive architecture. The core pipeline has been successfully migrated to LangGraph with significant improvements in code quality and performance.

## Current State (✅ Completed)

### Working Pipeline (4 LLM calls per decision)
1. **Generate memory queries** (~200 tokens) - Semantic query generation from observations
2. **Retrieve memories** (no LLM) - Vector search via ChromaDB
3. **Update cognitive state** (~500 tokens) - Merged memory report + working memory update
4. **Decide action** (~300 tokens) - Action selection based on cognitive/emotional state

**Total**: ~2000 tokens per decision cycle (~3.5s end-to-end with Gemini Flash Lite)

### Completed Features
- ✅ **LangGraph Migration** - Clean async pipeline with state management
- ✅ **Base Class Architecture** - Node & LLMNode eliminate boilerplate (29% code reduction)
- ✅ **Consolidated Memory Processing** - Single cognitive_update node replaces redundant steps
- ✅ **Instrumentation** - Automatic timing tracking, token usage tracking
- ✅ **Test Infrastructure** - Full test notebook (`notebooks/test_cognitive_pipeline.ipynb`)
- ✅ **Multiple Memory Retrieval** - ChromaDB returns top-k results per query
- ✅ **Prompt Variable Validation** - Runtime validation via `PROMPT_VARS` sets

### Known Issues
- ⚠️ **Memory Deduplication** - Duplicate memories sometimes retrieved (needs filtering)
- ⚠️ **No Memory Formation** - Old selective memory formation code removed, not yet reimplemented
- ⚠️ **Memories lack metadata** - No simulation time, location, or memory IDs
- ⚠️ **Unstructured working memory** - String format instead of structured cognitive state
- ⚠️ **Meta-cognitive prompts** - Still using "cognitive scientist" framing instead of direct embodiment

## Phase 1: Memory Enhancements (CURRENT PRIORITY)

### Goal
Add memory metadata, importance scoring, and conversation tracking to improve retrieval quality and reasoning depth.

### Deliverables

#### 1.1 Memory Metadata Enhancement
**Status:** ✅ Completed
**Priority:** High
**Spec:** [simulation_metadata_tool.md](backlog/simulation_metadata_tool.md)

Add contextual metadata to Memory model:
- ✅ `timestamp`: Simulation timestamp (game ticks/frames)
- ✅ `location`: Grid coordinates where event occurred
- ✅ `id`: Unique ID for citation (format: `memory_<uuid>`)
- ✅ Display format for LLM via `__str__()`: `[memory_<uuid> | T:<timestamp> | L:(x,y)] Memory content...`
- ✅ Created `VectorDBMetadata` model for type-safe metadata handling
- ✅ Created `VectorDBQuery` model to encapsulate search parameters
- ✅ Updated `ObservationContext` with `current_simulation_time` and `agent_location`
- ✅ Improved VectorDBMemory documentation and renamed parameters for clarity

**Benefits:**
- NPCs can reason about temporal/spatial context
- Enables memory citation in decision explanations
- Improves retrieval relevance through temporal/spatial filtering
- Strong typing throughout the memory system

#### 1.2 Memory Importance Scoring
**Status:** Not started
**Priority:** Medium

Implement importance-based retrieval weighting:
- Assign importance score (1-10) during memory formation
- Apply recency decay to older memories
- Calculate emotional weight from memory content
- Combined scoring: `relevance * importance * recency_factor`

**Implementation:**
- Add `calculate_importance()` method to [memory/store.py](../../src/mind/cognitive_architecture/memory/store.py)
- Modify retrieval to include importance in ranking

#### 1.3 Conversation History Tracking
**Status:** Not started
**Priority:** Medium

Track dialogue history separately from episodic memory:
- New conversation memory type with turn tracking
- Store speaker, content, timestamp
- Limited context window (last N turns)
- Separate retrieval path for social context

**Implementation:**
- New `memory/conversation_memory.py` module
- Integration with memory_retrieval node

#### 1.4 Memory Deduplication
**Status:** ✅ Completed
**Priority:** High (bug fix)

Filter duplicate memories from retrieval results:
- ✅ Added memory ID field to Memory model
- ✅ Created IdGenerator utility following Godot pattern
- ✅ Renamed MemoryStore → VectorDBMemory (modular component design)
- ✅ Added deduplication logic in [memory_retrieval/node.py](../../src/mind/cognitive_architecture/nodes/memory_retrieval/node.py)
- ✅ Uses memory IDs to detect duplicates
- ✅ Keeps first occurrence (preserves relevance ordering)

### Success Metrics
- Memory citations appear in action decisions
- Retrieval relevance improves with importance scoring
- Conversation context influences dialogue actions
- No duplicate memories in retrieval results

## Phase 2: Structured Cognitive State (NEXT)

### Goal
Replace unstructured working memory with flexible structured state and add memory formation.

### Deliverables

#### 2.1 Flexible Cognitive State
**Status:** Not started
**Priority:** High

Replace string-based working memory with structured but extensible state:
```json
{
  "current_situation": "Working at the forge, feeling focused",
  "active_goal": "Complete the sword commission",
  "recent_events": ["Started work", "Greeted apprentice"],
  "current_plan": ["Finish blade", "Polish", "Deliver"],
  "emotional_state": "content but slightly tired",
  // Additional fields as needed - not prescriptive
}
```

**Implementation:**
- Update [state.py](../../src/mind/cognitive_architecture/state.py) with new WorkingMemory model
- Modify cognitive_update node to output structured state
- Update prompts to work with structured format

#### 2.2 Selective Memory Formation
**Status:** Not started (regression from old code)
**Priority:** High

Re-implement conditional long-term memory formation:
- New pipeline node: `memory_formation`
- Trigger criteria: emotional intensity, goal relevance, novelty
- Importance scoring (1-10) during formation
- Only add significant experiences to long-term memory

**Implementation:**
- New `nodes/memory_formation/` module
- Add conditional edge in pipeline graph
- Importance calculation based on criteria

#### 2.3 Prompt Refinement
**Status:** Not started
**Priority:** Medium

Review and improve existing prompts:
- Consider direct embodiment vs. meta-cognitive framing
- Ensure consistency across all four nodes
- Add examples to improve output quality
- Document simulation mechanics clearly

### Success Metrics
- Cognitive state maintains coherence across decisions
- Memory formation reduces noise in memory store
- Prompt quality assessed through test scenarios

## Phase 3: Tool-Based Updates & Advanced Features (FUTURE)

### Goal
Enable selective state updates via tool calls and add planning/emotional depth.

### Deliverables

#### 3.1 Tool-Based Selective Updates
**Status:** Not started
**Priority:** Low (optimization)

Define tools for granular cognitive updates:
```python
def update_situation(new_situation: str):
    """Update current understanding of situation"""

def add_recent_event(event: str):
    """Add to recent events list"""

def update_plan(plan_modification: dict):
    """Modify current plan (add/remove/reorder steps)"""

def set_emotional_state(emotion: str):
    """Update emotional state"""
```

**Benefits:**
- Reduce token usage (update only changed fields)
- Improve state coherence (preserve unchanged fields)
- Better debugging (see exactly what changed)

#### 3.2 Enhanced Query Generation
**Status:** Not started
**Priority:** Medium

Improve memory query diversity:
- Use cognitive state to inform queries
- Contrastive queries (what's different/unexpected)
- Goal-relevant queries
- Temporal/situational similarity queries

#### 3.3 Planning System
**Status:** Not started
**Priority:** Medium
**Related Specs:** [basic_planning.md](backlog/basic_planning.md), [hierarchical_planning.md](backlog/hierarchical_planning.md)

Add multi-step planning capabilities:
- Plans as loose intentions (not rigid schedules)
- Natural language with optional structure
- Plan adaptation based on outcomes
- Multiple granularities without fixed hierarchy

#### 3.4 Emotional & Social Features
**Status:** Not started
**Priority:** Low
**Related Specs:** [emotional_memory.md](backlog/emotional_memory.md), [social_memory.md](backlog/social_memory.md), [theory_of_mind.md](backlog/theory_of_mind.md)

Advanced emotional and social modeling:
- Emotional state affects memory retrieval (mood-congruent recall)
- Relationship tracking in cognitive state
- Social context influences behavior
- Theory of mind for other NPCs
- Personality-driven cognitive patterns

### Success Metrics (Phase 3)
- Tool calls reduce token usage where appropriate
- Planning guides behavior without rigidity
- Emotional/social features feel natural, not forced

## Implementation Guidelines

### Design Principles
- **Flexibility over rigidity** - Let structure emerge from use
- **Simplicity first** - Add complexity only when needed
- **Type safety** - Strong typing with Pydantic models
- **Observability** - Instrument everything (timing, tokens, state changes)
- **Test-driven** - Test notebook for all major features

### Architecture Patterns
- **Base classes eliminate boilerplate** - Node & LLMNode handle common concerns
- **Prompts in markdown files** - Not hardcoded strings
- **Automatic instrumentation** - Metaclass magic for timing tracking
- **Validation at runtime** - PROMPT_VARS catches errors early

### Testing Strategy
- **Test notebook** - Primary development tool (`notebooks/test_cognitive_pipeline.ipynb`)
- **Unit tests** - Mock PipelineState, test individual nodes
- **Integration tests** - Full pipeline with real LLM calls
- **Performance profiling** - Monitor timing and token usage

## Technology Stack

### Core Dependencies
- **LangGraph** - Pipeline orchestration and state management
- **LangChain** - LLM abstraction layer (via OpenRouter)
- **ChromaDB** - Vector database for semantic memory retrieval
- **Pydantic** - Type-safe data models

### Why LangGraph
- First-class async and state management
- Graph structure fits cognitive pipeline naturally
- Excellent debugging with visual graph exploration
- Modern API with strong typing support

## Quick Reference

### Adding a New Pipeline Node
1. Create directory: `nodes/your_node/`
2. Create `prompt.md` with LLM instructions
3. Create `models.py` with Pydantic output model
4. Create `node.py`:
   ```python
   from ..base import LLMNode
   from .models import YourOutput

   class YourNode(LLMNode):
       step_name = "your_step"
       PROMPT_VARS = {"var1", "var2"}

       def __init__(self, llm):
           super().__init__(llm, output_model=YourOutput)

       async def process(self, state):
           output, tokens = await self.call_llm(
               var1=state.field1,
               var2=state.field2
           )
           state.your_output = output
           self.track_tokens(state, tokens)
           return state
   ```
5. Add to pipeline in [pipeline.py](../../src/mind/cognitive_architecture/pipeline.py)

### Configuration
- **API Key**: `credentials/api_keys.cfg` (format: `OPENROUTER_API_KEY:"{key}"`)
- **LLM Model**: Set in `get_llm()` call (see [langchain_llm.py](../../src/mind/apis/langchain_llm.py))
- **Memory Collection**: ChromaDB collection name in MemoryStore constructor

## Next Steps

**Immediate priorities (Phase 1):**
1. Memory metadata enhancement (simulation_time, location, memory_id)
2. Memory deduplication fix
3. Memory importance scoring
4. Conversation history tracking

**See [backlog/](backlog/) for detailed feature specifications.**