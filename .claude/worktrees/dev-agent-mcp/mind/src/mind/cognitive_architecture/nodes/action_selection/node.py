"""Action selection node implementation"""

import json
from pathlib import Path
from pprint import pformat

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.nodes.base import LLMNode
from mind.cognitive_architecture.observations import MindEvent, MindEventType
from mind.cognitive_architecture.state import PipelineState
from mind.knowledge import KnowledgeBase, KnowledgeFile
from mind.logging_config import get_logger

from .models import ActionSelectionOutput

logger = get_logger()


class ActionSelectionNode(LLMNode):
    """Models human-like action selection based on cognitive and emotional state"""

    step_name = "action_selection"

    def __init__(self, llm: BaseChatModel):
        # Load prompt template
        prompt_path = Path(__file__).parent / "prompt.md"
        prompt = PromptTemplate.from_template(prompt_path.read_text())

        super().__init__(llm, prompt, output_model=ActionSelectionOutput, max_retries=2)

    async def process(self, state: PipelineState) -> PipelineState:
        """Select an action based on current cognitive and emotional state"""

        # Format available actions
        actions_text = "\n".join([f"- {str(action)}" for action in state.available_actions])

        # Format personality traits
        personality_text = (
            ", ".join(state.personality_traits)
            if state.personality_traits
            else "No specific traits"
        )

        # Build world knowledge from centralized knowledge files
        world_knowledge = KnowledgeBase.get([
            KnowledgeFile.NEEDS,
            KnowledgeFile.INTERACTIONS,
            KnowledgeFile.MOVEMENT,
            KnowledgeFile.ACTIVITY,
        ])

        # Call LLM with prompt variables
        try:
            output = await self.call_llm(
                state,
                working_memory=str(state.working_memory),
                personality_traits=personality_text,
                available_actions=actions_text,
                recent_events=pformat(state.recent_events),
                world_knowledge=world_knowledge,
                format_instructions=self.get_format_instructions()
            )
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Action selection failed after retries, falling back to wait: {e}")
            # Use model_construct to bypass validation - WAIT is always safe
            fallback_action = Action.model_construct(action=ActionType.WAIT, parameters={})
            output = ActionSelectionOutput.model_construct(chosen_action=fallback_action)

        state.chosen_action = output.chosen_action

        # Create ACTION_CHOSEN event
        action_event = MindEvent(
            timestamp=state.observation.current_simulation_time,
            event_type=MindEventType.ACTION_CHOSEN,
            payload={
                "action": output.chosen_action.action,
                "parameters": output.chosen_action.parameters,
            }
        )
        state.recent_events.append(action_event)

        # Log action selection
        logger.debug(f"Evaluated {len(state.available_actions)} available actions")
        logger.debug(f"Selected: {output.chosen_action}")

        return state