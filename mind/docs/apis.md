# External APIs

## Overview

The APIs module provides LLM integration for agent decision-making via OpenRouter.

## LLM Client (`llm_client.py`)

### Configuration

Create `credentials/api_keys.cfg` with `OPENROUTER_API_KEY` to enable LLM access.

### Available Models

```python
class Model(Enum):
    HAIKU = "anthropic/claude-3-haiku"      # Default for agent operations
    SONNET = "anthropic/claude-3-5-sonnet"  # For future complex tasks
```

### Core Components

**Model.get_response()**: Sends prompts to LLM and returns text response.

**LLMFunction**: Combines a prompt template with a model for reusable operations.

```python
# Example usage in agents
memory_query_fn = LLMFunction(
    prompt=MemoryQueriesPrompt,
    model=Model.HAIKU
)
result = memory_query_fn.generate(
    working_memory="...",
    observation="..."
)
```

### Integration

Agents use LLMFunction instances for:
- Generating memory queries
- Creating memory reports  
- Updating working memory
- Selecting actions

All prompts are defined in `src/npc/prompts/` and parsed to extract structured data from LLM responses.