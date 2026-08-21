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
- `static_prefix: str | None = None` - Fully-formatted text prepended before every rendered prompt. Build it once with `LLMNode.build_static_prefix(template_text, **static_vars)`, which raises `ValueError` if the template declares a variable outside `static_vars` — anything above a cache breakpoint must be byte-identical across calls.
- `cache_control: bool = False` - Request a provider cache breakpoint after the static prefix

**Constraints**: If `max_retries > 0`, must provide `output_model`. If `cache_control`, must provide `static_prefix`.

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
- [reflection/node.py](../../../src/mind/cognitive_architecture/nodes/reflection/node.py) - With validation retry, salvage fallback, and prompt caching
- [memory_query/node.py](../../../src/mind/cognitive_architecture/nodes/memory_query/node.py) - Simple output

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

**Production Example**: [actions/models.py](../../../src/mind/cognitive_architecture/actions/models.py)

## Retry Mechanism

Each retry sends:
1. Previous AI response (as `AIMessage`)
2. Error message (as `HumanMessage`)

This gives the LLM context to fix its mistake. Tokens from all attempts are summed and tracked. Retries append after `messages[0]`, so the static prefix and its cache breakpoint survive every retry.

**Exhaustion contract**: after the last failed attempt, `call_llm` records the accumulated usage (the raise would otherwise escape past everything that can reach `PipelineState`), then either raises the last error (default) or, when the caller passed `on_exhausted=(raw_content, error) -> BaseModel`, returns that fallback's result instead. `ReflectionNode._salvage` is the production consumer: it rescues each output field independently from the last raw response. Transport errors are not routed through `on_exhausted` — there is nothing to salvage.

## Prompt Caching

When `cache_control` is set, the model is allowlisted (`constants.CACHE_CONTROL_MODELS`; unknown slugs default off so a provider that rejects the key cannot take the pipeline down), and the prefix clears `constants.MIN_CACHEABLE_PREFIX_CHARS`, message content is sent as two text blocks with the breakpoint on the first:

```python
[
    {"type": "text", "text": static_prefix, "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": dynamic_text},
]
```

Otherwise the same bytes go out as one plain string (`static_prefix + dynamic_text`). Test doubles (`AsyncMock`) have no string `model_name`, so caching is structurally off under test.

## Implementation Details

**Token Extraction**: `_extract_usage` reads the prompt/completion split from `usage_metadata` and answers cache accounting from the RAW provider dict at `response_metadata["token_usage"]` — the normalized `input_token_details` is always present in `langchain_openai >= 1.0` and proves nothing about the provider

**Validation Context**: Always `{"state": state}` - validators extract what they need from state

**Automatic Timing**: Inherited from `Node` base class, records in `state.time_ms[step_name]`

## Related Documentation

- [Node System](README.md) - Architecture and node catalog
- [reflection/node.py](../../../src/mind/cognitive_architecture/nodes/reflection/node.py) - Validation retry + salvage example
- [actions/models.py](../../../src/mind/cognitive_architecture/actions/models.py) - Context-aware validation patterns
