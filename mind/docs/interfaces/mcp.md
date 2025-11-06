# MCP Server

## Overview

The MCP server exposes the cognitive architecture over HTTP, enabling the Godot simulation to create NPC minds and request decisions based on structured observations. Communication uses the Model Context Protocol (MCP) with Server-Sent Events (SSE) transport.

## Architecture

```
MCP Server (server.py)
├── Mind Management
│   └── minds: Dict[str, Mind] - Active mind instances
├── Tools (RPC methods)
│   ├── create_mind - Initialize cognitive pipeline
│   ├── decide_action - Process observation → action
│   ├── consolidate_memories - Daily → long-term
│   └── cleanup_mind - Remove mind instance
└── Resources (State endpoints)
    ├── mind://{id}/state - Complete mental state
    ├── mind://{id}/working_memory - Current context
    └── mind://{id}/daily_memories - Unconsolidated memories
```

## Core Components

### MCPServer (`server.py`)

Manages active mind instances and exposes MCP tools for mind lifecycle and decision-making. Built on FastMCP with Starlette ASGI framework for SSE transport.

### Mind (`mind.py`)

Encapsulates a single mind's state and cognitive pipeline.

**Key Responsibilities:**
- Runs 4-node cognitive pipeline: memory query → retrieval → cognitive update → action selection
- Maintains working memory (current situational awareness)
- Buffers daily memories before consolidation to long-term storage
- Tracks conversation histories per interaction

**Configuration:** Initialized via `Mind.from_config(mind_id, config)` with `MindConfig` specifying entity ID, personality traits, LLM model, and memory storage path.

## MCP Tools

The server exposes four RPC methods for mind lifecycle and decision-making.

### create_mind

Creates a new mind instance with specified configuration. Takes `mind_id` and `MindConfig` (entity ID, traits, LLM model, memory storage path, optional seed memories).

### decide_action

Processes a structured observation through the cognitive pipeline and returns an action. Takes `mind_id` and `observation` dict containing entity status, needs, vision, and conversations. Validates observation structure, runs cognitive pipeline (query memories → retrieve → update working memory → select action), and returns action dict or error.

### consolidate_memories

Moves daily memories from buffer to long-term ChromaDB storage. Typically called at natural break points like sleep or scene transitions.

### cleanup_mind

Removes a mind instance and frees associated resources.

## MCP Resources

Read-only endpoints for inspecting mind state:

- `mind://{mind_id}/state` - Complete mental state (traits, working memory, memory counts, active conversations)
- `mind://{mind_id}/working_memory` - Current situational awareness
- `mind://{mind_id}/daily_memories` - Unconsolidated memories pending storage

## Integration with Godot

The Godot simulation connects through a layered client system:

**Godot → Python Flow:**
1. `McpMindClient` (GDScript) - Collects observations from simulation
2. `McpSdkClient` (C#) - Marshals data and manages WebSocket
3. `McpServiceProxy` (C#) - MCP protocol communication
4. **MCP Server** (this component) - Validates and processes
5. **Cognitive Pipeline** - Generates decision
6. Response flows back through layers to simulation

**Key Data Transformations:**
- `CompositeObservation.get_data()` → observation dict
- Observation dict → validated `Observation` Pydantic model
- `Action.model_dump()` → action dict for Godot

## HTTP Endpoints

Beyond MCP protocol endpoints, the server provides standard HTTP endpoints for lifecycle management and monitoring.

### /health (GET)

Returns server status and uptime. Used by clients to detect when the server is ready after startup, avoiding failed connection attempts during initialization.

**Integration:** Godot client polls this endpoint until receiving 200 OK before attempting MCP connection.

### /shutdown (POST)

Triggers graceful server shutdown via HTTP request instead of OS process signals. This enables cross-platform server management without handling Windows/WSL process boundaries, where killing wrapper processes can leave Python processes orphaned.

### /logs (GET)

Returns structured log entries for client-side display and debugging. Stores the most recent 1000 log entries in memory with timestamp, level, and message.

**Query Parameters:**
- `since` (optional): Unix timestamp - returns only logs after this time for incremental fetching
- `limit` (optional): Maximum entries to return (default: 100)

**Response Format:**
```json
{
  "logs": [
    {"timestamp": 1736096823.456, "level": "INFO", "message": "Server started"},
    {"timestamp": 1736096824.123, "level": "DEBUG", "message": "SSE client connected"}
  ]
}
```

**Integration:** Game clients poll this endpoint every 1-2 seconds to fetch new logs for developer console display alongside simulation logs.

## Running the Server

### Basic Usage

```bash
cd /home/hearn/projects/npc/mind
poetry run python -m mind.interfaces.mcp.main
```

Server starts on `localhost:8000` by default.

### Command-Line Arguments

```bash
# Custom host and port
python -m mind.interfaces.mcp.main --host 127.0.0.1 --port 8000

# View all options
python -m mind.interfaces.mcp.main --help
```

### Configuration

Set `OPENROUTER_API_KEY` environment variable for LLM access.

### Server Output

```
Starting NPC Mind MCP server on http://127.0.0.1:8000/sse
```

## Error Handling

Tools return errors with `status: "error"` and descriptive `error_message`. Common failures include invalid mind IDs, malformed observations, LLM API errors, and ChromaDB storage issues.

## Development

**Adding Tools:**

Define async methods with `@self.mcp.tool()` decorator. FastMCP handles serialization via type hints. Access active minds through `self.minds` dictionary.

**Testing:**

Integration tests in `tests/integration/test_mcp_server.py` verify the complete workflow: create mind → decide action → consolidate memories → cleanup. Run via `poetry run pytest tests/integration/test_mcp_server.py -v`.

**Debugging:**

Common issues include observation validation failures (check FastMCP errors for field mismatches), LLM API errors (verify OPENROUTER_API_KEY and rate limits), and memory storage problems (monitor ChromaDB logs).
