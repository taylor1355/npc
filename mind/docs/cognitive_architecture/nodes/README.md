# Cognitive Pipeline Nodes

**Location**: `src/mind/cognitive_architecture/nodes/`

The cognitive pipeline processes observations through specialized nodes, each performing a specific cognitive function.

## Node Architecture

### Base Classes

**Node** ([base.py](../../src/mind/cognitive_architecture/nodes/base.py))
- Base class for all pipeline nodes
- Provides automatic execution timing via `__init_subclass__` decorator
- Subclasses implement `async def process(state) -> state`

**LLMNode** ([base.py](../../src/mind/cognitive_architecture/nodes/base.py))
- Extends `Node` for nodes that call LLMs
- Handles retry logic with error feedback
- Tracks token usage across retry attempts
- Supports validation with pipeline state context
- See [LLMNode documentation](base.md) for details

### Node Hierarchy

```
Node (automatic timing)
├── LLMNode (retry + token tracking + validation)
│   ├── MemoryQueryNode
│   ├── CognitiveUpdateNode
│   └── ActionSelectionNode
└── (Direct Node extensions)
    ├── MemoryRetrievalNode
    └── MemoryConsolidationNode
```

## Pipeline Flow

```
1. MemoryQueryNode        → Generate semantic queries
2. MemoryRetrievalNode    → Fetch memories from vector store
3. CognitiveUpdateNode    → Update working memory, form new memories
4. ActionSelectionNode    → Select validated action
5. MemoryConsolidationNode → (Separate: daily → long-term transfer)
```

## Node Catalog

### Memory Nodes

- **[MemoryQueryNode](memory_query.md)** - Generates diverse semantic queries for memory retrieval
- **[MemoryRetrievalNode](memory_retrieval.md)** - Searches vector store and deduplicates results
- **[MemoryConsolidationNode](memory_consolidation.md)** - Transfers daily memories to long-term storage

### Decision Nodes

- **[CognitiveUpdateNode](cognitive_update.md)** - Updates working memory, situation assessment, goals, emotional state
- **[ActionSelectionNode](action_selection.md)** - Selects actions with context-aware validation and retry

## State Management

**PipelineState Fields:**
- `observation: Observation` - Current sensory input
- `working_memory: WorkingMemory` - Situational context
- `retrieved_memories: list[Memory]` - Fetched long-term memories
- `daily_memories: list[NewMemory]` - Unconsolidated experiences
- `chosen_action: Action | None` - Selected action
- `tokens_used: dict[str, int]` - Token counts per node
- `time_ms: dict[str, int]` - Execution time per node

## Performance Tracking

Both base classes automatically track metrics in pipeline state:
- **Timing**: `Node` tracks execution time in `state.time_ms[step_name]`
- **Tokens**: `LLMNode` tracks token usage in `state.tokens_used[step_name]`

## Creating Custom Nodes

### Extending Node

For data processing without LLM calls:

```python
class MyDataNode(Node):
    step_name = "my_step"

    async def process(self, state: PipelineState) -> PipelineState:
        # Process data
        state.result = transform(state.data)
        return state  # Timing tracked automatically
```

**Example**: [memory_retrieval/node.py](../../src/mind/cognitive_architecture/nodes/memory_retrieval/node.py)

### Extending LLMNode

For nodes that call LLMs:

```python
class MyLLMNode(LLMNode):
    step_name = "my_step"

    def __init__(self, llm):
        prompt = PromptTemplate.from_template(...)
        super().__init__(llm, prompt, output_model=MyOutput, max_retries=2)

    async def process(self, state):
        output = await self.call_llm(state, data=str(state.data))
        state.result = output.result
        return state  # Tokens + timing tracked automatically
```

**Examples**:
- [action_selection/node.py](../../src/mind/cognitive_architecture/nodes/action_selection/node.py) - With validation retry
- [memory_query/node.py](../../src/mind/cognitive_architecture/nodes/memory_query/node.py) - Simple query generation

## Testing

Unit tests mock LLMs with `AsyncMock` and verify behavior.

**Test Examples**:
- [test_base_node.py](../../tests/unit/test_base_node.py) - Base class testing patterns
- [test_action_selection_node.py](../../tests/unit/test_action_selection_node.py) - Node-specific tests
- [test_memory_query_node.py](../../tests/unit/test_memory_query_node.py) - LLM mocking patterns

## Best Practices

1. **Single Responsibility** - One cognitive function per node
2. **External Prompts** - Store prompts in `.md` files
3. **Type Safety** - Use Pydantic models for LLM outputs
4. **Validation** - Implement `@model_validator` for context-aware checks
5. **Testing** - Mock LLMs and test retry behavior

## Related Documentation

- [LLMNode Details](base.md) - Retry logic, validation context, token tracking
- [Actions System](../actions/README.md) - Action types and validation
- [Pipeline Architecture](../pipeline.md) - How nodes compose
