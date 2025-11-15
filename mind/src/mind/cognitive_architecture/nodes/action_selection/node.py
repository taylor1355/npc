"""Action selection node implementation"""

from pathlib import Path
from pprint import pformat

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from mind.cognitive_architecture.nodes.base import LLMNode
from mind.cognitive_architecture.state import PipelineState
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

        # Call LLM with prompt variables
        output = await self.call_llm(
            state,
            working_memory=str(state.working_memory),
            personality_traits=personality_text,
            available_actions=actions_text,
            recent_events=pformat(state.recent_events),
            format_instructions=self.get_format_instructions()
        )

        state.chosen_action = output.chosen_action

        # Log action selection
        logger.debug(f"Evaluated {len(state.available_actions)} available actions")
        logger.debug(f"Selected: {output.chosen_action}")

        return state