# Cognitive Architecture Development Roadmap

**Last Updated:** October 10, 2025

---

## Current State

### What's Working
- LangGraph-based cognitive pipeline (4 nodes: query → retrieve → update → decide)
- Memory system with metadata, importance scoring, and deduplication
- Structured working memory (Pydantic models)
- MCP server with tools and resources
- Test coverage (integration tests + notebook)

### Known Issues
- Meta-cognitive prompt framing ("You are a cognitive psychologist...") reduces immersion
- End-to-end integration with npc-simulation repo needs verification

---

## Near-term Priorities (Next 1-2 Weeks)

### 1. End-to-End Integration Testing
**Why:** Need to verify mind + npc-simulation work together
**Effort:** ~1-2 days
**Priority:** HIGH

**Tasks:**
- Start MCP server (`src/mind/interfaces/mcp/main.py`)
- Launch npc-simulation Godot project
- Verify bidirectional communication
- Test full decision cycle: observation → action → execution
- Document any integration issues found

**Success Criteria:**
- NPCs in simulation make decisions via cognitive pipeline
- No crashes or communication errors
- Performance is acceptable (<5s per decision)

### 2. Prompt Refinement
**Why:** Meta-cognitive framing reduces character immersion
**Effort:** ~1-2 days
**Priority:** MEDIUM

**Tasks:**
- Review 3 prompt files (memory_query, cognitive_update, action_selection)
- Rewrite for direct embodiment (first-person perspective)
- Remove "cognitive psychologist" framing
- Add concrete examples to each prompt
- Test behavior changes in notebook

**Files to update:**
- `src/mind/cognitive_architecture/nodes/memory_query/prompt.md`
- `src/mind/cognitive_architecture/nodes/cognitive_update/prompt.md`
- `src/mind/cognitive_architecture/nodes/action_selection/prompt.md`

**Success Criteria:**
- No meta-cognitive framing remains
- Consistent voice across all nodes
- NPCs feel more "inhabited" in testing

### 3. Conversation History Formalization (Optional)
**Why:** Social interactions could benefit from better memory
**Effort:** ~2-3 days
**Priority:** LOW (only if social gameplay needs depth)

**Current State:**
- Mind class tracks `conversation_histories` dict
- Works but not integrated into memory retrieval

**Tasks:**
- Decide: conversation memory vs episodic memory?
- Add conversation-specific retrieval path
- Implement context window (last N turns)
- Test conversation coherence

**Success Criteria:**
- NPCs reference past conversations appropriately
- No memory bloat from routine chat

---

## Backlog (Future Work)

Prioritized by **obviousness × development velocity × concreteness** (see [planning/README.md](README.md))

### Planning System
**Impact:** HIGH velocity (unlocks behavioral chunking, goal-driven behavior)
**Concreteness:** HIGH (immediately visible in gameplay)
**Specs:** [basic_planning.md](backlog/basic_planning.md), [hierarchical_planning.md](backlog/hierarchical_planning.md)

Multi-timescale behavioral coherence:
- Daily routines → weekly goals → long-term aspirations
- Plans as loose intentions, not rigid schedules
- Plan adaptation based on outcomes

### Enhanced Memory Retrieval
**Impact:** MEDIUM velocity (improves quality, not architecture)
**Concreteness:** MEDIUM (subtle gameplay improvements)
**Spec:** [generative_agents_memory.md](backlog/generative_agents_memory.md)

Improvements to memory search:
- Contrastive queries (what's different/unexpected?)
- Goal-relevant memory weighting
- Reflection generation (higher-order memories)

### Emotional & Social Intelligence
**Impact:** LOW velocity (adds complexity without enabling new systems)
**Concreteness:** HIGH (very noticeable in gameplay)
**Specs:** [emotional_memory.md](backlog/emotional_memory.md), [social_memory.md](backlog/social_memory.md), [theory_of_mind.md](backlog/theory_of_mind.md)

Advanced social modeling:
- Mood-congruent recall
- Relationship tracking
- Theory of mind (modeling other NPCs' beliefs)
- Emotional contagion

### Tool-Based Selective Updates
**Impact:** NEUTRAL velocity (optimization, doesn't unlock features)
**Concreteness:** NONE (invisible to players)

Token usage optimization:
- Update only changed fields in working memory
- Reduce token costs without changing behavior
- Only prioritize if costs become prohibitive

---

## Architecture Reference

### Pipeline Structure
```
1. Memory Query (LLM)
   ↓
2. Memory Retrieval (vector search)
   ↓
3. Cognitive Update (LLM) - updates WorkingMemory, forms new memories
   ↓
4. Action Selection (LLM)
```

### Key Models
- **PipelineState** - Data flowing through nodes
- **WorkingMemory** - Structured cognitive state (situation, goals, plan, emotions)
- **Memory** - Long-term memory with metadata (id, content, timestamp, location, importance)
- **Observation** - Structured input from simulation (status, needs, vision, conversations)
- **Action** - Chosen action with type and parameters

### Adding a New Node
1. Create directory: `nodes/your_node/`
2. Create `prompt.md` with LLM instructions
3. Create `models.py` with Pydantic output model
4. Create `node.py` extending LLMNode
5. Add to pipeline in `pipeline.py`

See "Quick Reference" section in previous roadmap version for code template.

### Testing
- **Integration tests:** `tests/test_mcp_integration.py`
- **Test notebook:** `notebooks/test_cognitive_pipeline.ipynb`
- **Instrumentation:** Automatic timing/token tracking via Node base class

---

## Design Principles

From [planning/README.md](README.md):

1. **Obviousness** - Do obvious things first to constrain less obvious things
2. **Development Velocity** - How does this affect speed of future development?
3. **Concreteness** - Does this directly improve player experience?

Guidelines:
- **Flexibility over rigidity** - Let structure emerge from use
- **Simplicity first** - Add complexity only when needed
- **Type safety** - Strong typing with Pydantic models
- **Observability** - Instrument everything

---

## Completed Milestones (Reference)

### Phase 0: Godot Integration (Oct 2025)
- Mind class with cognitive pipeline + memory store
- Structured Observation model
- MCP server with tools (create_mind, decide_action, consolidate_memories, cleanup_mind)
- Integration tests passing

### Phase 1: Memory Enhancements (Oct 2025)
- Memory metadata (timestamp, location, unique ID)
- Importance scoring (1-10)
- Recency decay in retrieval
- Deduplication by ID

### Phase 2: Structured Cognitive State (Oct 2025)
- WorkingMemory Pydantic model
- Selective memory formation
- Daily memory buffer with consolidation
- Extensible schema (`extra = "allow"`)

---

**For detailed feature specifications, see [backlog/](backlog/)**
