# MCP Server

## Overview

The MCP (Model Context Protocol) server provides network access to the NPC cognitive architecture. It exposes agent management and decision-making capabilities through a standardized protocol, enabling the Godot simulation to communicate with NPC agents over HTTP/SSE.

## Architecture

```
MCP Server
├── MCPServer Class - Server initialization
├── AgentManager - Agent lifecycle management
│   └── agents: Dict[str, Agent] - Active agents
├── Tools (RPC methods)
│   ├── create_agent - Initialize new NPCs
│   ├── process_observation - Generate actions
│   └── cleanup_agent - Remove NPCs
└── Resources (Data endpoints)
    └── agent://{id}/info - Agent state
```

## Server Components

### MCPServer Class (`mcp_server.py`)

The main server class that registers tools and resources:

**Initialization:**
```python
server = MCPServer(name="NPC Manager")
# Uses FastMCP internally for protocol handling
# Binds to 0.0.0.0 by default (accepts any IP)
```

**Transport:**
- Uses Server-Sent Events (SSE) for communication
- HTTP endpoint: `/sse` for connections
- Runs on Starlette ASGI framework

### AgentManager Class

Manages the lifecycle of NPC agents:

**Key Methods:**
- `create_agent(agent_id, config)`: Initialize new agent
- `get_agent(agent_id)`: Retrieve existing agent
- `cleanup_agent(agent_id)`: Remove agent and free resources

**Agent Storage:**
- Maintains dictionary of active agents
- Each agent has unique string ID (from Godot instance ID)
- Agents persist until explicitly cleaned up

## MCP Tools

### create_agent

Creates a new NPC agent with specified configuration.

**Parameters:**
- `agent_id`: Unique identifier string
- `config`: Dictionary with:
  - `traits`: List of personality traits
  - `initial_working_memory`: Starting context (optional)
  - `initial_long_term_memories`: List of past experiences (optional)

**Returns:** Status and agent_id

### process_observation

Processes an observation and returns the chosen action.

**Parameters:**
- `agent_id`: Agent to process for
- `observation`: Natural language description of current state
- `available_actions`: List of actions, each with:
  - `name`: Action identifier
  - `description`: Human-readable description
  - `parameters`: Action-specific parameters

**Returns:** Selected action name and empty parameters dict

**Processing Flow:**
1. Retrieve agent by ID
2. Format actions for LLM (indexed descriptions)
3. Call agent.process_observation()
4. Return chosen action name and empty parameters

### cleanup_agent

Removes an agent and cleans up resources.

**Parameters:**
- `agent_id`: Agent identifier to remove

**Returns:** Status and agent_id

## MCP Resources

### agent://{agent_id}/info

Retrieves current agent state information.

**URL Pattern:** `agent://npc_123/info`

**Returns:** JSON with status, traits, and working_memory

## Running the Server

### Command Line Options

```bash
# Default configuration
poetry run python -m npc.mcp_server

# Custom host/port
poetry run python -m npc.mcp_server --host 0.0.0.0 --port 8080
```

**Arguments:**
- `--host`: IP address to bind (default: 127.0.0.1)
- `--port`: Port number (default: 8000)

### Configuration Requirements

Create `credentials/api_keys.cfg` with `OPENROUTER_API_KEY` for LLM access.

### Server Output

```
Starting NPC MCP server on http://127.0.0.1:8000/sse
```

## Integration with Simulation

The Godot simulation connects through a multi-layer client system:

1. **McpNpcClient** (GDScript): High-level interface
2. **McpSdkClient** (C#): Data marshalling bridge  
3. **McpServiceProxy** (C#): MCP SDK connection
4. **MCP Server** (this component): Agent management

### Communication Protocol

**Connection:**
- Client connects to `/sse` endpoint
- Maintains persistent SSE connection
- Handles reconnection on failure

**Message Flow:**
- Client sends tool calls with parameters
- Server processes and returns responses
- All communication uses JSON format

## Error Handling

All tools return errors in standard format:

```json
{
  "status": "error",
  "message": "Agent npc_123 not found"
}
```

Common errors:
- Agent not found
- Invalid configuration
- LLM processing failures
- Connection timeouts

## Performance Considerations

- Agents remain in memory until cleanup
- Each observation triggers LLM calls
- ChromaDB persists to disk
- SSE maintains open connections

## Development Notes

**Adding New Tools:**
1. Define tool function with `@mcp.tool()` decorator
2. Add parameters with type hints
3. Return dictionary response
4. Handle exceptions gracefully

**Testing:**
- Use MCP client libraries for integration tests
- Mock LLM responses for unit tests
- Monitor memory usage with many agents