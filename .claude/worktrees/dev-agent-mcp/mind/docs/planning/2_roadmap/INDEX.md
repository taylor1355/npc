# Cognitive Architecture Development Roadmap

**Last Updated:** November 28, 2025

---

## Current State

### What's Working
- LangGraph-based cognitive pipeline (4 nodes: query → retrieve → update → decide)
- Memory system with metadata, importance scoring, and deduplication
- Structured working memory (Pydantic models)
- MCP server with tools and resources
- Centralized knowledge system for simulation context
- Test coverage (unit tests + integration tests + notebook)
- Event-driven temporal decision-making
- Action validation with retry logic

### Recent Improvements (Nov 2025)
- Reorganized codebase for locality of behavior
- Improved action selection prompts (removed meta-cognitive framing)
- Added simulation mechanics context to reduce state fixation
- Fixed entity hallucination via validation + retry
- Added HTTP lifecycle endpoints and logging infrastructure

---

## Near-term Priorities

### Basic Planning System
**Why:** NPCs make moment-to-moment decisions without broader structure
**Impact:** HIGH velocity (unlocks goal-driven behavior)
**Spec:** [basic_planning.md](../3_backlog/basic_planning.md)

Enable NPCs to maintain daily routines and work toward goals rather than purely reactive behavior.

### Sleep as Cognitive Resource
**Why:** Natural integration point for memory consolidation
**Impact:** HIGH (consolidation + planning reset point)
**Spec:** [sleep_as_cognitive_resource.md](../3_backlog/sleep_as_cognitive_resource.md)

Use sleep cycles to trigger memory consolidation and daily plan generation.

---

## Backlog Highlights

See [3_backlog/INDEX.md](../3_backlog/INDEX.md) for full categorized list.

**High Priority:**
- [Cluster-Based Memory Consolidation](../3_backlog/cluster_based_memory_consolidation.md) - Smarter memory organization
- [MCP Debugging](../3_backlog/mcp_debugging.md) - Mental state visibility for development

**Medium Priority:**
- [Emotional Modeling](../3_backlog/emotional_modeling.md) - Mood affects behavior
- [Personality](../3_backlog/personality.md) - Individual behavioral consistency

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

### Testing
- **Unit tests:** `tests/unit/`
- **Integration tests:** `tests/test_mcp_integration.py`
- **Test notebook:** `notebooks/test_cognitive_pipeline.ipynb`

---

## Design Principles

From [README.md](../README.md):

1. **Obviousness** - Do obvious things first to constrain less obvious things
2. **Development Velocity** - How does this affect speed of future development?
3. **Concreteness** - Does this directly improve player experience?

Guidelines:
- **Flexibility over rigidity** - Let structure emerge from use
- **Simplicity first** - Add complexity only when needed
- **Type safety** - Strong typing with Pydantic models
- **Observability** - Instrument everything
