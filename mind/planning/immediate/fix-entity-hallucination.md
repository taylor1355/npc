# Fix Entity Hallucination Problem

## Problem Statement

NPCs are hallucinating entity IDs and interaction names that don't exist in their observations, causing "Target not found" errors.

**Example**:
- **Observation shows**: `entity_55b585f3-7068-4e97-a219-c5f61d9c402c` with interaction `sit`
- **LLM returns**: `chair_001` with interaction `rest`
- **Result**: Action fails, target not found

## Root Causes Identified

1. **CRITICAL**: `available_actions` was always empty - LLM saw NO actions to choose from
2. **CRITICAL**: Vision data didn't show entity IDs/interaction names in observation string
3. **HIGH**: Prompt examples used simplified IDs ("tom_001") vs real UUIDs
4. **HIGH**: No validation or retry when LLM returns invalid actions

## Solution Strategy

### Phase 1: Provide Complete Information to LLM ✅ DONE
1. ✅ Created `Observation.get_available_actions()` - builds action list from observation
2. ✅ Populated `state.available_actions` in MCP server
3. ✅ Updated `Observation.__str__()` to show entity IDs and interaction details

### Phase 2: Validate Actions with Retry + Error Feedback (IN PROGRESS)

**Goal**: Validate LLM output against observation and retry with error feedback if invalid.

**Approach**: Pydantic validation + Custom RetryParser

#### Why This Approach?

- **Validation belongs in the model** - Action validates itself using Pydantic
- **LangChain retry handles error feedback** - Automatically injects errors into retry prompts
- **Need validation context** - Action needs Observation to validate against
- **RetryWithErrorOutputParser doesn't support context** - So we subclass it

#### Implementation Plan

**1. Move validation logic into Action model**

File: `actions/models.py`

```python
from pydantic import BaseModel, Field, model_validator, ValidationInfo

class Action(BaseModel):
    """Action to be executed"""

    model_config = {"use_enum_values": True}

    action: ActionType
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_executable(self, info: ValidationInfo):
        """Validate action can be executed given observation.

        Requires 'observation' in validation context.
        Raises ValidationError if action is invalid.
        """
        # Fail loudly if no context - don't silently skip validation
        if not info.context:
            raise ValueError("Action validation requires context with 'observation'")

        observation = info.context.get('observation')
        if not observation:
            raise ValueError("Action validation requires 'observation' in context")

        # Validation logic (moved from validation.py)
        self._validate_movement_lock(observation)

        if self.action == ActionType.INTERACT_WITH:
            self._validate_interact_with(observation)
        elif self.action == ActionType.MOVE_TO:
            self._validate_move_to()

        return self

    def _validate_movement_lock(self, observation):
        """Check if movement-based actions are blocked"""
        from .exceptions import MovementLockedError

        if observation.status and observation.status.movement_locked:
            if self.action in (ActionType.MOVE_TO, ActionType.MOVE_DIRECTION, ActionType.WANDER):
                raise MovementLockedError()

    def _validate_interact_with(self, observation):
        """Validate INTERACT_WITH action"""
        from .exceptions import InvalidEntityError, InvalidInteractionError, MissingRequiredParameterError

        entity_id = self.parameters.get("entity_id")
        interaction_name = self.parameters.get("interaction_name")

        if not entity_id:
            raise MissingRequiredParameterError("entity_id", self.action)
        if not interaction_name:
            raise MissingRequiredParameterError("interaction_name", self.action)

        if observation.vision:
            visible_ids = [e.entity_id for e in observation.vision.visible_entities]
            if entity_id not in visible_ids:
                raise InvalidEntityError(entity_id, visible_ids)

            entity = next(e for e in observation.vision.visible_entities if e.entity_id == entity_id)
            if interaction_name not in entity.interactions:
                raise InvalidInteractionError(interaction_name, entity_id, list(entity.interactions.keys()))

    def _validate_move_to(self):
        """Validate MOVE_TO action"""
        from .exceptions import MissingRequiredParameterError

        if "destination" not in self.parameters:
            raise MissingRequiredParameterError("destination", self.action)
```

**2. Create custom retry parser with validation context support**

File: `nodes/base.py` (or new `nodes/retry_parser.py`)

```python
from langchain.output_parsers import RetryWithErrorOutputParser
from pydantic import ValidationError
import json

class RetryWithValidationContext(RetryWithErrorOutputParser):
    """Extends RetryWithErrorOutputParser to support Pydantic validation context.

    This allows models to validate against external context (e.g., Observation)
    while still getting automatic retry with error feedback.
    """

    def __init__(self, parser, llm, validation_context: dict | None = None, max_retries: int = 2):
        super().__init__(parser=parser, llm=llm, max_retries=max_retries)
        self.validation_context = validation_context or {}

    def parse(self, completion: str) -> Any:
        """Parse with validation context passed to Pydantic"""
        try:
            # Parse JSON from completion
            data = json.loads(completion)

            # Validate with context - this triggers model_validator
            return self.parser.pydantic_object.model_validate(
                data,
                context=self.validation_context
            )
        except (json.JSONDecodeError, ValidationError) as e:
            # Parent class handles retry with error feedback
            raise e

    def set_validation_context(self, context: dict):
        """Update validation context (useful for per-request context)"""
        self.validation_context = context
```

**3. Use in ActionSelectionNode**

File: `nodes/action_selection/node.py`

```python
from langchain_core.messages import HumanMessage
from mind.cognitive_architecture.nodes.retry_parser import RetryWithValidationContext

class ActionSelectionNode(LLMNode):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, output_model=ActionSelectionOutput)

        # Create retry parser (without context initially)
        self.retry_parser = RetryWithValidationContext(
            parser=self.parser,
            llm=llm,
            max_retries=2  # 1 initial + 2 retries = 3 total
        )

    async def process(self, state: PipelineState) -> PipelineState:
        # Format inputs...
        actions_text = "\n".join([f"- {str(action)}" for action in state.available_actions])
        # ... context_text, personality_text

        # Format prompt
        prompt = self.prompt_template.format(
            working_memory=str(state.working_memory),
            cognitive_context=context_text if context_text else "No specific context",
            personality_traits=personality_text,
            available_actions=actions_text,
            format_instructions=self.parser.get_format_instructions()
        )

        # Call LLM
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        # Set validation context for this request
        self.retry_parser.set_validation_context({'observation': state.observation})

        # Parse with retry (automatic error feedback on validation failure)
        output = await self.retry_parser.aparse_with_prompt(
            completion=response.content,
            prompt_value=messages[0]
        )

        # Track tokens
        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        self.track_tokens(state, tokens)

        state.chosen_action = output.chosen_action
        return state
```

**4. Remove old validation.py (logic moved to Action model)**

File: `actions/validation.py` - DELETE

The validation logic now lives in `Action.validate_executable()` and helper methods.

**5. Keep exceptions.py as-is**

File: `actions/exceptions.py` - NO CHANGES

These are still raised from Action's validation methods.

## Benefits of This Approach

1. ✅ **Validation is part of the model** - Action owns its validation logic
2. ✅ **Automatic retry with error feedback** - LangChain handles it
3. ✅ **Validation context support** - Custom parser passes context to Pydantic
4. ✅ **Reusable pattern** - RetryWithValidationContext can be used by other nodes
5. ✅ **Fails loudly** - No silent validation skipping
6. ✅ **Clean separation** - Validation logic in model, retry logic in parser

## Files to Change

1. **actions/models.py** - Add `@model_validator` with validation logic
2. **nodes/retry_parser.py** - NEW: RetryWithValidationContext class
3. **nodes/action_selection/node.py** - Use RetryWithValidationContext
4. **actions/validation.py** - DELETE (logic moved to Action model)

## Testing Plan

1. Unit test Action validation with various invalid cases
2. Test retry behavior with mock LLM returning invalid actions
3. Verify error messages are injected into retry prompts
4. Integration test with real observation data
5. Run full test suite

## Success Criteria

- ✅ No more "Target not found" errors from hallucinated IDs
- ✅ LLM sees exact entity IDs and interaction names
- ✅ Invalid actions trigger retry with clear error feedback
- ✅ System retries up to 3 times before failing
- ✅ All tests pass

## Related Files

- Root cause analysis: (documented in initial investigation)
- Refactor commit: File structure reorganization (completed)
