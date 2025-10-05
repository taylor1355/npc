# NPC Cognitive Architecture

This repository implements the cognitive architecture for NPCs, providing memory management and decision-making capabilities via LLM integration. It serves as the "brain" component for the [NPC simulation system](https://github.com/taylor1355/npc-simulation).

## Quick Start

### Installation

```bash
poetry install
```

### Configuration

Create `credentials/api_keys.cfg` with:
```
OPENROUTER_API_KEY:"{your-api-key}"
```

### Run MCP Server

```bash
# Start the agent server
poetry run python -m npc.mcp_server

# The simulation connects to this server for NPC decisions
```

## Architecture Overview

- **Agent System**: Memory management and LLM-based decisions
- **MCP Server**: Network protocol for remote agent control  
- **Simulators**: Test environments for agent behavior

See [docs/](docs/) for detailed documentation.

## Development

```bash
# Run tests
poetry run pytest
```