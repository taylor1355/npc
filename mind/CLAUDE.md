# NPC System - Claude Code Context

## Project Overview

This repository provides the **Mind tier** cognitive architecture for the NPC simulation system. It runs as an MCP (Model Context Protocol) server that provides LLM-based decision-making and memory management as a service to the Godot simulation.

## Sister Project

The Godot-based simulation is located at `/mnt/c/Users/hearn/Dev/gamedev/npc-simulation` on WSL.

The sister project contains the visual simulation environment (Controller and Client tiers) where these NPC cognitive architectures (Mind tier) operate. Refer to its documentation at `/mnt/c/Users/hearn/Dev/gamedev/npc-simulation/docs` for:
- Three-tier architecture overview (Controller → Client → Mind)
- Scene architecture and entity systems
- MCP integration patterns and protocol details
- Visual components and UI

## Architecture

**Current State - In Transition:**

```
Mind MCP Server
├── cognitive_architecture/ - LangGraph pipeline (active development)
│   ├── pipeline.py - StateGraph orchestration
│   ├── nodes/ - Processing nodes
│   │   ├── memory_query/ - Generate memory queries
│   │   ├── memory_retrieval/ - Retrieve relevant memories
│   │   ├── cognitive_update/ - Update working memory
│   │   └── action_selection/ - Choose actions
│   ├── memory/store.py - Vector memory backend
│   └── state.py - Pipeline state management
├── mcp_server.py - MCP service layer (currently uses legacy Agent)
├── agent/ - Legacy prototype code (being replaced)
│   └── agent.py - Early memory + decision prototype
├── apis/ - External service integrations
│   └── llm_client.py - OpenRouter via OpenAI client
└── simulators/ - Test environments
    └── text_adventure/ - Standalone demo
```

**Migration Status:**
- The `agent.py` was early prototyping and is being replaced
- New LangGraph-based cognitive pipeline is under active development
- MCP server currently uses legacy Agent class, will migrate to new pipeline
- See `docs/architecture/cognitive_architecture_design.md` for vision and roadmap

## Key Commands

- `poetry run python -m mind.mcp_server` - Start MCP server (default: localhost:8000)
- `poetry run python -m mind.mcp_server --host 0.0.0.0 --port 8080` - Custom host/port
- `poetry run pytest` - Run tests

## MCP Server

The server exposes cognitive architecture via MCP protocol:

**Tools:**
- `create_agent(agent_id, config)` - Initialize NPC with traits and memories
- `decide_action(agent_id, observation, available_actions)` - Process observation and return action
- `cleanup_agent(agent_id)` - Remove agent and cleanup resources

**Resources:**
- `agent://{id}/mental_state` - Query agent's mental state, memory, and personality

## Development Focus

This project focuses on NPC intelligence and behavior as a service. The cognitive architecture integrates with the Godot simulation through MCP, providing the "Mind" tier of the three-tier architecture.