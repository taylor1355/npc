# NPC Mind - Claude Code Context

## Project Overview

This repository provides the **Mind tier** cognitive architecture for the NPC simulation system. It runs as an MCP (Model Context Protocol) server that provides LLM-based decision-making and memory management as a service to the Godot simulation.

**Status:** Production-ready core architecture with LangGraph pipeline, structured memory system, and full MCP integration.

## Sister Project

The Godot-based simulation is located at `/mnt/c/Users/taylor/Dev/gamedev/npc-simulation` on WSL.

The sister project contains the visual simulation environment (Controller and Client tiers) where these NPC cognitive architectures (Mind tier) operate. Refer to its documentation for:
- Three-tier architecture overview (Controller → Client → Mind)
- Scene architecture and entity systems
- MCP integration patterns and protocol details
- Visual components and UI

## Architecture

```
src/mind/
├── cognitive_architecture/          # LangGraph-based pipeline
│   ├── pipeline.py                  # StateGraph orchestration
│   ├── state.py                     # PipelineState model
│   ├── models.py                    # Shared models (Memory, Observation, Action)
│   ├── nodes/                       # Processing nodes
│   │   ├── base.py                  # Node & LLMNode base classes
│   │   ├── memory_query/            # Generate semantic search queries
│   │   │   ├── node.py
│   │   │   ├── models.py
│   │   │   └── prompt.md
│   │   ├── memory_retrieval/        # Vector search via ChromaDB
│   │   │   └── node.py
│   │   ├── cognitive_update/        # Update working memory, form memories
│   │   │   ├── node.py
│   │   │   ├── models.py            # WorkingMemory, NewMemory
│   │   │   └── prompt.md
│   │   ├── action_selection/        # Choose action based on context
│   │   │   ├── node.py
│   │   │   ├── models.py
│   │   │   └── prompt.md
│   │   └── memory_consolidation/    # Daily → long-term storage
│   │       └── node.py
│   └── memory/
│       └── vector_db_memory.py      # ChromaDB backend
├── interfaces/mcp/                  # MCP server interface
│   ├── server.py                    # FastMCP server
│   ├── mind.py                      # Mind runtime state
│   ├── models.py                    # MindConfig, validation
│   └── main.py                      # Server entry point
├── apis/                            # LLM integration
│   ├── langchain_llm.py             # Primary LLM client (OpenRouter)
│   ├── llm_client.py                # Direct OpenAI client
│   └── messages.py                  # Message utilities
└── prompts/                         # Legacy prompt templates (archived)
```

### Cognitive Pipeline (4-step LangGraph)

1. **Memory Query** (~200 tokens) - Generate semantic search queries from observations
2. **Memory Retrieval** (no LLM) - ChromaDB vector search with deduplication
3. **Cognitive Update** (~500 tokens) - Update WorkingMemory, form new memories
4. **Action Selection** (~300 tokens) - Choose action based on full context

**Performance:** ~2000 tokens, ~3.5s end-to-end (Gemini Flash Lite)

### Memory System

- **Vector storage** via ChromaDB with semantic search
- **Metadata:** timestamp, location, unique ID
- **Importance scoring:** 1-10 scale via LLM evaluation
- **Deduplication:** ID-based filtering in retrieval
- **Daily buffer:** Consolidates to long-term on demand
- **Recency decay:** Applied in retrieval scoring

### Working Memory (Structured State)

```python
class WorkingMemory(BaseModel):
    situation_assessment: str = ""
    active_goals: list[str] = []
    recent_events: list[str] = []
    current_plan: list[str] = []
    emotional_state: str = ""

    class Config:
        extra = "allow"  # Extensible for future fields
```

## MCP Server

### Tools

- **`create_mind(mind_id, config)`** - Initialize cognitive pipeline with traits and seed memories
- **`decide_action(mind_id, observation)`** - Process structured observation → action dict
- **`consolidate_memories(mind_id)`** - Transfer daily memories → long-term storage
- **`cleanup_mind(mind_id)`** - Remove mind instance and free resources

### Resources

- **`mind://{id}/state`** - Complete mental state snapshot
- **`mind://{id}/working_memory`** - Current working memory (JSON)
- **`mind://{id}/daily_memories`** - Unconsolidated memories list

### Integration Flow

```
Godot (GDScript)
  → McpMindClient (serializes CompositeObservation)
    → McpSdkClient (C#, manages WebSocket)
      → MCP Server (Python, FastMCP)
        → Mind.decide_action()
          → CognitivePipeline.process()
            → Returns Action dict
              → Flows back to Godot for execution
```

## Key Commands

### Server
```bash
# Start MCP server (default: localhost:3000)
poetry run python src/mind/interfaces/mcp/main.py

# With environment variable for API key
export OPENROUTER_API_KEY="your-key"
poetry run python src/mind/interfaces/mcp/main.py
```

### Testing
```bash
# Run all tests
poetry run pytest

# Integration tests only
poetry run pytest tests/test_mcp_integration.py -v

# Interactive development
jupyter notebook notebooks/test_cognitive_pipeline.ipynb
```

### Development
```bash
# Install dependencies
poetry install

# Check code structure
tree src/mind/cognitive_architecture/nodes/
```

## Current Development Focus

See [docs/planning/roadmap.md](docs/planning/roadmap.md) for detailed planning.

**Near-term priorities:**
1. End-to-end integration testing with npc-simulation repo
2. Prompt refinement (remove meta-cognitive framing)
3. Conversation history formalization (optional)

**Future features:**
- Planning system (multi-timescale behavioral coherence)
- Enhanced memory retrieval (contrastive queries, reflections)
- Emotional & social intelligence (mood-congruent recall, theory of mind)

## Documentation

- **[docs/README.md](docs/README.md)** - Overview and architecture
- **[docs/cognitive_architecture/overview.md](docs/cognitive_architecture/overview.md)** - Vision and design philosophy
- **[docs/interfaces/mcp.md](docs/interfaces/mcp.md)** - MCP protocol details
- **[docs/planning/roadmap.md](docs/planning/roadmap.md)** - Development roadmap and priorities
- **[docs/planning/backlog/](docs/planning/backlog/)** - Feature specifications

## Design Principles

- **Flexibility over rigidity** - Let structure emerge from use
- **Simplicity first** - Add complexity only when needed
- **Type safety** - Strong typing with Pydantic models throughout
- **Observability** - Automatic timing/token tracking via base classes
- **Test-driven** - Comprehensive test coverage (integration + notebook)

## Technology Stack

- **LangGraph** - Async pipeline orchestration
- **LangChain** - LLM abstraction (via OpenRouter)
- **ChromaDB** - Vector database for semantic memory
- **Pydantic** - Type-safe data models
- **FastMCP** - MCP protocol implementation
- **Poetry** - Dependency management
