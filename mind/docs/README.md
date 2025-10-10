# NPC Cognitive Architecture Documentation

## Overview

This repository provides the cognitive architecture for NPCs used in the [NPC simulation](https://github.com/taylor1355/npc-simulation). It implements mind-based decision making through a LangGraph cognitive pipeline with memory systems and LLM integration, exposed via the Model Context Protocol (MCP) for network communication.

## Architecture

```
NPC Cognitive Architecture
├── Cognitive Pipeline (cognitive_architecture/)
│   ├── Memory Query - Generate semantic search queries
│   ├── Memory Retrieval - Fetch relevant long-term memories
│   ├── Cognitive Update - Update working memory, form new memories
│   └── Action Selection - Choose action based on context
├── MCP Server (interfaces/mcp/)
│   ├── Tool: create_mind - Initialize cognitive pipeline
│   ├── Tool: decide_action - Process observation → action
│   ├── Tool: consolidate_memories - Daily → long-term storage
│   ├── Tool: cleanup_mind - Remove mind instance
│   └── Resources: mind://{id}/* - State endpoints
└── Memory Systems
    ├── Working Memory - Pydantic model for current context
    ├── Daily Memories - Buffer for unconsolidated experiences
    └── Long-term Memory - ChromaDB vector store
```

## Core Components

### Cognitive Architecture

The cognitive pipeline (`src/mind/cognitive_architecture/`) processes observations through a 4-node async pipeline:

**Memory Systems:**
- **Working Memory**: Pydantic model tracking situation assessment, active goals, current plan, emotional state
- **Daily Memories**: Buffer of experiences with importance scores, consolidated at natural break points
- **Long-term Memory**: ChromaDB vector store enabling semantic retrieval of past experiences

**Processing Pipeline:**
1. **Memory Query**: LLM generates semantic search queries for relevant memories
2. **Memory Retrieval**: Fetches top-k similar memories from ChromaDB
3. **Cognitive Update**: Updates working memory with new context, forms new daily memories
4. **Action Selection**: Chooses action from available options based on full context

**Key Models:**
- `Observation`: Structured observation with status, needs, vision, conversations
- `WorkingMemory`: Current situational context with flexible schema
- `Action`: Chosen action with type and parameters

### MCP Server

The MCP server (`src/mind/interfaces/mcp/`) provides network access to minds via Server-Sent Events:

**Mind Management:**
- `Mind` class encapsulates cognitive pipeline, memory store, conversation histories
- `Mind.from_config()` creates instances with specified traits and initial memories
- Dictionary-based registry manages active mind instances

**Tools:**
- `create_mind(mind_id, config)`: Initialize mind with cognitive pipeline
- `decide_action(mind_id, observation)`: Process structured observation → action dict
- `consolidate_memories(mind_id)`: Transfer daily → long-term storage
- `cleanup_mind(mind_id)`: Remove mind and free resources

**Resources:**
- `mind://{id}/state`: Complete mental state snapshot
- `mind://{id}/working_memory`: Current working memory as JSON
- `mind://{id}/daily_memories`: Unconsolidated memories list

### Integration with Godot

The Godot simulation communicates through a layered client system:

**Communication Flow:**
1. **EntityController** (Godot): Collects `CompositeObservation` with structured sub-observations
2. **McpMindClient** (GDScript): Serializes observation via `get_data()` → dict
3. **McpSdkClient** (C#): Manages WebSocket connection and request/response correlation
4. **MCP Server** (Python): Validates dict → `Observation` model, runs cognitive pipeline
5. **Response**: Action dict flows back through layers → Godot executes

**Observation Structure:**
Structured data including entity_id, simulation_time, status (position, movement), needs (hunger, energy, etc.), vision (visible entities with interactions), conversations (per-interaction histories).

**Action Types:**
- `move_to`: Navigate to specific position
- `interact_with`: Engage with entity (item or NPC)
- `wander`: Random movement
- `wait`: Remain idle
- `act_in_interaction`: Interaction-specific actions
- `cancel_interaction`: End current interaction
- `respond_to_interaction_bid`: Accept/reject interaction requests

## Setup

### Dependencies
```bash
poetry install
```

### Configuration

Set environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key"
```

### Running the Server
```bash
# Start MCP server (default: localhost:3000)
poetry run python src/mind/interfaces/mcp/main.py
```

## Development

### Testing
```bash
# Run integration tests
poetry run python -m pytest tests/test_mcp_integration.py -v

# All tests
poetry run pytest
```

### Project Structure
```
src/mind/
├── cognitive_architecture/    # Pipeline and memory systems
│   ├── nodes/                 # Pipeline nodes
│   └── memory/                # Memory implementations
├── interfaces/mcp/            # MCP server and models
├── apis/                      # LLM client integration
└── prompts/                   # LLM prompt templates
```

## Documentation

- [cognitive_architecture/overview.md](cognitive_architecture/overview.md) - Detailed pipeline architecture
- [interfaces/mcp.md](interfaces/mcp.md) - MCP protocol and tools
- [planning/roadmap.md](planning/roadmap.md) - Development roadmap
- [meta/style-guide.md](meta/style-guide.md) - Documentation standards

## Related Resources

- Godot simulation: [npc-simulation](https://github.com/taylor1355/npc-simulation)
- Development guide: [CLAUDE.md](/CLAUDE.md)
