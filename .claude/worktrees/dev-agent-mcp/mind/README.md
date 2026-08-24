# NPC Cognitive Architecture

This repository implements the cognitive architecture for NPCs, providing LLM-based decision-making and memory management via the Model Context Protocol (MCP). It serves as the "Mind tier" for the [NPC simulation system](https://github.com/taylor1355/npc-simulation).

## Quick Start

### Installation

```bash
poetry install
```

### Configuration

Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key"
```

### Run MCP Server

```bash
# Start the Mind MCP server (default: localhost:3000)
poetry run python src/mind/interfaces/mcp/main.py
```

The Godot simulation connects to this server for NPC cognitive processing.

## What This Provides

**Cognitive Pipeline:** 4-step async processing via LangGraph
1. Generate memory queries from observations
2. Retrieve relevant memories via vector search
3. Update working memory and form new memories
4. Select actions based on full cognitive context

**Memory System:** Vector-based storage with semantic search, importance scoring, and temporal/spatial metadata

**MCP Integration:** FastMCP server exposing tools and resources for mind management

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         Godot Simulation (Client)       │
└──────────────┬──────────────────────────┘
               │ MCP Protocol
               ↓
┌─────────────────────────────────────────┐
│         Mind MCP Server (Python)        │
│  ┌───────────────────────────────────┐  │
│  │   Cognitive Pipeline (LangGraph)  │  │
│  │   ├─ Memory Query                 │  │
│  │   ├─ Memory Retrieval             │  │
│  │   ├─ Cognitive Update             │  │
│  │   └─ Action Selection             │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │   Memory System (ChromaDB)        │  │
│  │   - Vector storage                │  │
│  │   - Semantic search               │  │
│  │   - Metadata (time, location, ID) │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## MCP Tools

- **`create_mind(mind_id, config)`** - Initialize cognitive pipeline
- **`decide_action(mind_id, observation)`** - Process observation → action
- **`consolidate_memories(mind_id)`** - Move daily → long-term storage
- **`cleanup_mind(mind_id)`** - Remove mind instance

## Documentation

- **[docs/README.md](docs/README.md)** - Comprehensive overview and architecture details
- **[docs/planning/roadmap.md](docs/planning/roadmap.md)** - Development roadmap and priorities
- **[docs/cognitive_architecture/overview.md](docs/cognitive_architecture/overview.md)** - Vision and design philosophy
- **[docs/interfaces/mcp.md](docs/interfaces/mcp.md)** - MCP protocol details
- **[CLAUDE.md](CLAUDE.md)** - Claude Code development context

## Development

### Testing

```bash
# Run all tests
poetry run pytest

# Integration tests only
poetry run pytest tests/test_mcp_integration.py -v

# Interactive development notebook
jupyter notebook notebooks/test_cognitive_pipeline.ipynb
```

### Adding Features

See [docs/planning/roadmap.md](docs/planning/roadmap.md) for current priorities and [docs/planning/backlog/](docs/planning/backlog/) for detailed feature specifications.

## Technology Stack

- **LangGraph** - Async pipeline orchestration
- **LangChain** - LLM abstraction layer (via OpenRouter)
- **ChromaDB** - Vector database for semantic memory
- **Pydantic** - Type-safe data models
- **FastMCP** - MCP protocol implementation
- **Poetry** - Dependency management
