# NPC Cognitive Architecture Documentation

## Overview

This repository provides the cognitive architecture for NPCs used in the [NPC simulation module](https://github.com/taylor1355/npc-simulation). It implements agent-based decision making through memory systems and LLM integration, exposed via the Model Context Protocol (MCP) for network communication.

## Architecture

```
NPC Cognitive Architecture
├── Agent System (agent.py)
│   ├── Working Memory - Recent context and observations
│   ├── Long-term Memory - Vector database (ChromaDB)
│   └── Decision Engine - LLM-based action selection
├── MCP Server (mcp_server.py)
│   ├── Tool: create_agent - Initialize new NPCs
│   ├── Tool: process_observation - Generate actions
│   ├── Tool: cleanup_agent - Remove NPCs
│   └── Resource: agent://{id}/info - Agent state
└── Simulators
    ├── Base Interface (base_interface.py) - Simulator protocol
    └── Text Adventure (text_adventure.py) - Standalone demo
```

## Core Components

### Agent System

The `Agent` class (`src/npc/agent/agent.py`) manages NPC cognition:

**Memory Architecture:**
- **Working Memory**: Maintains recent observations and current context as text
- **Long-term Memory**: ChromaDB vector store for semantic retrieval of past experiences
- **Memory Updates**: Each observation triggers memory queries and working memory updates

**Decision Process:**
1. Receive observation from simulator
2. Query long-term memory for relevant experiences
3. Update working memory with new context
4. Generate action decision via LLM
5. Store significant observations in long-term memory

### MCP Server

The MCP server (`src/npc/mcp_server.py`) provides network access to agents:

**Tools:**
- `create_agent(agent_id, config)`: Initialize agent with traits and memories
- `process_observation(agent_id, observation, available_actions)`: Generate action choice
- `cleanup_agent(agent_id)`: Remove agent and free resources

**Resources:**
- `agent://{agent_id}/info`: Retrieve agent state (traits, working memory)

The server runs on HTTP/SSE transport, typically on port 8000.

### Integration with Simulation

The simulation module communicates with this cognitive architecture through a three-tier system:

1. **NpcController** (GDScript): Gathers observations every 3 seconds
2. **McpNpcClient** (GDScript/C# bridge): Formats observations and sends to MCP
3. **MCP Server** (this repo): Processes observations and returns actions

**Observation Format:**
The simulation sends structured observations including:
- NPC needs (hunger, hygiene, fun, energy) as percentages
- Visible entities with available interactions
- Current state and interaction status
- Recent events (interaction lifecycle, conversations)

**Action Types:**
The cognitive architecture can return:
- `move_to`: Navigate to specific location
- `interact_with`: Engage with item or NPC
- `wander`: Random movement
- `wait`: Remain idle
- `act_in_interaction`: Perform interaction-specific actions
- `cancel_interaction`: End current interaction

## Setup

### Dependencies
```bash
poetry install
```

### Configuration

Create `credentials/api_keys.cfg`:
```
OPENROUTER_API_KEY:"{your-api-key}"
```

### Running the Server
```bash
# Start MCP server (default: localhost:8000)
poetry run python -m npc.mcp_server

# Custom host/port
poetry run python -m npc.mcp_server --host 0.0.0.0 --port 8080
```

## Development

### Testing
```bash
poetry run pytest
```

### Extending Behavior

**Modify Prompts**: Edit templates in `src/npc/prompts/` to change:
- Memory query generation
- Working memory updates
- Action decision logic

**Add Memory Types**: Extend `MemoryDatabase` to support:
- Different embedding models
- Additional metadata
- Custom retrieval strategies

**Create Simulators**: Implement `BaseSimulator` interface for new environments.

## Documentation

- [agent.md](agent.md) - Detailed agent architecture
- [mcp-server.md](mcp-server.md) - MCP protocol and tools
- [simulators.md](simulators.md) - Environment implementations
- [apis.md](apis.md) - External service integrations

## Related Resources

- Simulation module: [$NPC](https://github.com/taylor1355/npc-simulation)
- Development guide: [CLAUDE.md](/CLAUDE.md)
- Style guide: [meta/style-guide.md](meta/style-guide.md)