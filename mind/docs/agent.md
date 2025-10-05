# Agent System

## Overview

The Agent system provides the cognitive architecture for NPCs, managing their memory, personality, and decision-making processes. Each agent maintains both working and long-term memory, using LLMs to process observations and generate contextually appropriate actions.

## Architecture

```
Agent
├── Memory Systems
│   ├── Working Memory (text) - Current context
│   └── Long-term Memory (ChromaDB) - Past experiences
├── LLM Functions
│   ├── Query Generator - Find relevant memories
│   ├── Memory Report Generator - Summarize context
│   ├── Working Memory Generator - Update context
│   ├── Long-term Memory Generator - Extract key events
│   └── Action Decision Generator - Choose actions
└── Personality
    └── Traits (list[str]) - Behavioral modifiers
```

## Core Components

### Agent Class (`agent.py`)

The `Agent` class orchestrates all cognitive processes:

**Initialization:**
```python
Agent(
    llm_config: AgentLLMConfig,
    initial_working_memory: str = "",
    initial_long_term_memories: list[str] = [],
    personality_traits: list[str] = []
)
```

**Key Attributes:**
- `working_memory`: Current context and recent observations (string)
- `long_term_memory`: MemoryDatabase instance for semantic retrieval
- `personality_traits`: List of traits influencing behavior
- `current_observation`: Latest observation from simulator
- `current_actions`: Available actions from simulator

### Memory Database (`memory_database.py`)

The `MemoryDatabase` provides semantic memory storage using ChromaDB:

**Key Methods:**
- `add(text: str)`: Store new memory with embedding
- `retrieve(query: str, top_k: int = 5)`: Find similar memories
- `__len__()`: Count stored memories

**Implementation Details:**
- Uses sentence transformers for embeddings
- Stores memories in ChromaDB collection
- Retrieves by semantic similarity

## Processing Pipeline

### Observation Processing

When `process_observation()` is called:

1. **Update State**: Store observation and available actions
2. **Update Working Memory**: 
   - Generate queries based on observation
   - Retrieve relevant long-term memories
   - Create memory report summarizing context
   - Update working memory text
3. **Update Long-term Memory**: Extract and store key information
4. **Choose Action**: Generate decision based on full context

### Memory Update Process

**Query Generation:**
```python
# Generate 3-5 queries to find relevant memories
queries = query_generator.generate(
    working_memory=current_memory,
    observation=new_observation
)
```

**Memory Retrieval:**
```python
# Retrieve top matches for each query
for query in queries:
    memories = long_term_memory.retrieve(query, top_k=1)
```

**Working Memory Update:**
```python
# Create report and update working memory
report = memory_report_generator.generate(
    working_memory=current_memory,
    observation=observation,
    retrieved_memories=memories
)
new_memory = working_memory_generator.generate(
    working_memory=current_memory,
    memory_report=report
)
```

### Action Selection

The action decision process considers:
- Current observation
- Working memory context
- Personality traits
- Available actions with descriptions

**Decision Generation:**
```python
decision = action_decision_generator.generate(
    working_memory=memory,
    observation=observation,
    available_actions=formatted_actions,
    personality_traits=traits
)
return decision["action_index"]
```

## Prompt Templates

All LLM interactions use structured prompts from `src/npc/prompts/`:

### Key Templates

**Memory Queries** (`memory_queries_template.py`):
- Generates 3-5 search queries
- Focuses on entities, locations, activities
- Returns JSON list of queries

**Memory Report** (`memory_report_template.py`):
- Summarizes working memory and retrieved memories
- Highlights relevant context
- 2-3 paragraph narrative

**Working Memory** (`working_memory_template.py`):
- Updates or replaces working memory
- Maintains recent context
- Structured as observation log

**Action Decision** (`action_decision_template.py`):
- Analyzes situation and personality
- Evaluates each available action
- Returns action index with reasoning

## Configuration

### LLM Configuration

```python
@dataclass
class AgentLLMConfig:
    small_llm: Model  # For most operations
    large_llm: Model  # For complex decisions (future use)
```

### Supported Models

Via `llm_client.py`:
- Claude models (Haiku, Sonnet, Opus)
- GPT models (3.5, 4)
- Open models via OpenRouter

## Usage Example

```python
# Initialize agent
config = AgentLLMConfig(
    small_llm=Model.HAIKU,
    large_llm=Model.SONNET
)
agent = Agent(
    llm_config=config,
    personality_traits=["curious", "friendly"],
    initial_working_memory="Just arrived at the market."
)

# Process observation
observation = "You see a merchant selling apples and a guard standing nearby."
actions = {
    0: "approach merchant",
    1: "talk to guard", 
    2: "continue walking"
}
chosen_action = agent.process_observation(observation, actions)
```

## Integration Notes

The Agent system is designed to work with:
- **MCP Server**: Manages agent lifecycle and network access
- **Simulators**: Provide observations and execute actions
- **Memory Storage**: Persists to disk via ChromaDB

See [mcp-server.md](mcp-server.md) for network integration details.