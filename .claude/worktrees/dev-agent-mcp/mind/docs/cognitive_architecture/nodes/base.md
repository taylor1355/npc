# LLMNode Base Class

**Location**: `src/mind/cognitive_architecture/nodes/base.py`

Base class for pipeline nodes that call LLMs. Handles retry logic, token tracking, and validation context automatically.

## Core Features

**Automatic Retry with Error Feedback**
When validation fails, LLMNode appends the error message to the conversation and retries. The LLM sees its mistake and can correct it.

**Token Tracking Across Retries**
Accumulates token usage from all attempts (including failures) in `state.tokens_used[step_name]`.

**Validation Context**
Passes `{"state": state}` to Pydantic validators, enabling context-aware validation (e.g., validating actions against available options).

**Dual Output Modes**
Supports Pydantic models (structured output with validation) or raw strings (no parsing).

## Configuration

**Required**:
- `llm: BaseChatModel` - LangChain chat model
- `prompt: PromptTemplate` - Prompt template with variables

**Optional**:
- `output_model: type[BaseModel] | None = None` - Pydantic model (None for raw strings)
- `max_retries: int = 0` - Retry attempts on validation failure

**Constraints**: If `max_retries > 0`, must provide `output_model`

## Usage

```python
class MyNode(LLMNode):
    step_name = "my_step"

    def __init__(self, llm: BaseChatModel):
        prompt = PromptTemplate.from_template(Path(__file__).parent / "prompt.md").read_text())
        super().__init__(llm, prompt, output_model=MyOutput, max_retries=2)

    async def process(self, state: PipelineState) -> PipelineState:
        output = await self.call_llm(state, data=state.data)
        state.result = output.result
        return state  # Tokens + timing tracked automatically
```

**Production Examples**:
- [action_selection/node.py](../../src/mind/cognitive_architecture/nodes/action_selection/node.py) - With validation retry
- [memory_query/node.py](../../src/mind/cognitive_architecture/nodes/memory_query/node.py) - Simple output

## Context-Aware Validation

Validators receive pipeline state:

```python
class MyOutput(BaseModel):
    chosen_item: str

    @model_validator(mode='after')
    def validate_against_state(self, info: ValidationInfo):
        state = info.context.get('state')
        if self.chosen_item not in state.observation.available_items:
            raise ValueError(f"{self.chosen_item} not available")
        return self
```

**Production Example**: [actions/models.py](../../src/mind/cognitive_architecture/actions/models.py)

## Retry Mechanism

Each retry sends:
1. Previous AI response (as `AIMessage`)
2. Error message (as `HumanMessage`)

This gives the LLM context to fix its mistake. Tokens from all attempts are summed and tracked.

## Implementation Details

**Token Extraction**: Extracts from `response.usage_metadata['total_tokens']` (works with all LangChain models)

**Validation Context**: Always `{"state": state}` - validators extract what they need from state

**Automatic Timing**: Inherited from `Node` base class, records in `state.time_ms[step_name]`

## Related Documentation

- [Node System](README.md) - Architecture and node catalog
- [Action Selection](action_selection.md) - Validation retry example
- [Actions](../actions/README.md) - Context-aware validation patterns
