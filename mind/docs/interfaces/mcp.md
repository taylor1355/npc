# MCP Server

## Overview

The MCP server provides network access to the cognitive architecture, exposing mind management and decision-making capabilities through the Model Context Protocol. It enables the Godot simulation to communicate with NPC minds using structured observations and actions.

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

### MCPServer Class (`src/mind/interfaces/mcp/server.py`)

Main server class that manages mind instances and registers MCP tools.

**Initialization:**
```python
server = MCPServer(name="NPC Mind Server")
# Uses FastMCP for protocol handling
# Registers tools and resources on construction
```

**Transport:**
- Server-Sent Events (SSE) over HTTP
- Connects to `/sse` endpoint
- Built on Starlette ASGI framework

### Mind Class (`src/mind/interfaces/mcp/mind.py`)

Encapsulates a single mind's runtime state and cognitive pipeline.

**Key Components:**
- `CognitivePipeline`: 4-node async processing (memory query → retrieval → update → action selection)
- `VectorDBMemory`: ChromaDB-backed long-term storage
- `WorkingMemory`: Pydantic model for current situational context
- `conversation_histories`: Per-interaction dialogue tracking
- `daily_memories`: Buffer for unconsolidated experiences

**Creation Pattern:**
```python
Mind.from_config(mind_id, config)
# Initializes LLM, memory store, seeds initial memories
# Returns ready-to-use Mind instance
```

### MindConfig (`src/mind/interfaces/mcp/models.py`)

Configuration for creating minds.

**Fields:**
- `entity_id: str` - Simulation entity identifier
- `traits: list[str]` - Personality traits for decision-making
- `llm_model: str` - Model ID (default: "google/gemini-2.0-flash-lite-001")
- `embedding_model: str` - For memory retrieval (default: "all-MiniLM-L6-v2")
- `memory_storage_path: str` - ChromaDB persistence location
- `initial_working_memory: WorkingMemory | None` - Starting context
- `initial_long_term_memories: list[str]` - Seed memories

## MCP Tools

### create_mind

Initializes a new mind with cognitive pipeline and memory storage.

**Parameters:**
- `mind_id: str` - Unique identifier
- `config: MindConfig` - Initialization configuration

**Returns:** `{status: "created", mind_id: str}`

**Process:**
1. Creates Mind instance via `Mind.from_config()`
2. Seeds long-term memories if provided
3. Initializes working memory
4. Stores in active minds dictionary

### decide_action

Processes structured observation through cognitive pipeline and returns chosen action.

**Parameters:**
- `mind_id: str` - Mind to process for
- `observation: dict` - Structured observation data (validated to `Observation` model)

**Observation Structure:**
```python
{
    "entity_id": str,
    "current_simulation_time": int,  # Game minutes
    "status": {
        "position": tuple[int, int],
        "movement_locked": bool,
        "current_interaction": dict,
        "controller_state": dict
    },
    "needs": {
        "needs": dict[str, float],
        "max_value": float
    },
    "vision": {
        "visible_entities": [EntityData]
    },
    "conversations": [ConversationObservation]
}
```

**Returns:** `{status: str, action: dict | None, error_message: str | None}`

**Processing Flow:**
1. Validate observation dict to `Observation` Pydantic model
2. Update conversation histories (aggregate by interaction_id)
3. Build `PipelineState` with observation, working memory, traits
4. Run through 4-node cognitive pipeline:
   - Memory query: Generate search query
   - Memory retrieval: Fetch relevant long-term memories
   - Cognitive update: Update working memory, form new memories
   - Action selection: Choose action based on context
5. Update mind's working memory and daily memories
6. Return action dict for simulation execution

### consolidate_memories

Transfers daily memories to long-term storage via embedding and ChromaDB indexing.

**Parameters:**
- `mind_id: str` - Mind to consolidate

**Returns:** `{status: "success", consolidated_count: int}`

**Process:**
1. Runs memory consolidation node on daily memories buffer
2. Clears daily memories after successful storage
3. Typically called at natural break points (sleep, scene transitions)

### cleanup_mind

Removes mind instance and frees resources.

**Parameters:**
- `mind_id: str` - Mind to remove

**Returns:** `{status: "removed", mind_id: str}`

## MCP Resources

### mind://{mind_id}/state

Returns complete mental state snapshot.

**Returns:**
```python
{
    "entity_id": str,
    "traits": list[str],
    "working_memory": WorkingMemory,
    "daily_memories_count": int,
    "long_term_memory_count": int,
    "active_conversations": list[str]
}
```

### mind://{mind_id}/working_memory

Returns current working memory as JSON.

### mind://{mind_id}/daily_memories

Returns list of unconsolidated daily memories.

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

## Running the Server

### Basic Usage

```bash
cd /home/hearn/projects/npc/mind
poetry run python src/mind/interfaces/mcp/main.py
```

Server starts on `localhost:3000` by default.

### Configuration

Set `OPENROUTER_API_KEY` environment variable for LLM access.

### Server Output

```
MCP Server running on http://localhost:3000
```

## Error Handling

All tools return errors in consistent format:

```python
{
    "status": "error",
    "action": None,
    "error_message": "Mind not found" | "Invalid observation format: ..."
}
```

Common error cases:
- Mind not found (invalid mind_id)
- Observation validation failure (malformed data)
- LLM processing errors (API failures, rate limits)
- Memory store errors (ChromaDB issues)

## Performance Characteristics

- **Latency**: 100ms-2s per decision (depends on LLM model)
- **Memory**: Each mind holds working memory, daily buffer, conversation histories
- **Storage**: ChromaDB persists to disk at `memory_storage_path`
- **Connections**: SSE maintains persistent connections per client
- **Concurrency**: Handles multiple concurrent decide_action calls

## Development Notes

**Adding New Tools:**
1. Define async function with `@self.mcp.tool()` decorator
2. Use type hints for FastMCP auto-serialization
3. Return dict (FastMCP handles JSON conversion)
4. Access `self.minds` dictionary for mind instances

**Testing:**
- Integration tests: `tests/test_mcp_integration.py`
- Run via: `poetry run python -m pytest tests/test_mcp_integration.py -v`
- Tests verify create → decide → consolidate → cleanup workflow

**Debugging:**
- Enable debug logging in Godot client for request/response visibility
- Check FastMCP validation errors for malformed observations
- Monitor ChromaDB logs for memory storage issues
