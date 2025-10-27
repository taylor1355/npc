"""Action selection node implementation"""

from langchain_core.language_models import BaseChatModel

from ...state import PipelineState
from ..base import LLMNode
from .models import ActionSelectionOutput

# Cognitive context keys (shared with cognitive_update)
KEY_SITUATION = "situation_assessment"
KEY_GOALS = "current_goals"
KEY_EMOTIONAL = "emotional_state"


class ActionSelectionNode(LLMNode):
    """Models human-like action selection based on cognitive and emotional state"""

    step_name = "action_selection"
    PROMPT_VARS = {"working_memory", "cognitive_context", "personality_traits", "available_actions"}

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, output_model=ActionSelectionOutput)

    async def process(self, state: PipelineState) -> PipelineState:
        """Select an action based on current cognitive and emotional state"""

        # Format available actions using their __str__ methods
        actions_text = "\n".join([f"- {str(action)}" for action in state.available_actions])

        # Format cognitive context
        context_text = ""
        if state.cognitive_context:
            if KEY_SITUATION in state.cognitive_context:
                context_text += f"Situation: {state.cognitive_context[KEY_SITUATION]}\n"
            if KEY_GOALS in state.cognitive_context:
                goals = state.cognitive_context[KEY_GOALS]
                if goals:
                    context_text += f"Current Goals: {', '.join(goals)}\n"
            if KEY_EMOTIONAL in state.cognitive_context:
                context_text += f"Emotional State: {state.cognitive_context[KEY_EMOTIONAL]}"

        personality_text = (
            ", ".join(state.personality_traits)
            if state.personality_traits
            else "No specific traits"
        )

        # Select action
        output, tokens = await self.call_llm(
            working_memory=str(state.working_memory),
            cognitive_context=context_text if context_text else "No specific context",
            personality_traits=personality_text,
            available_actions=actions_text,
        )

        # Update state
        state.chosen_action = output.chosen_action
        self.track_tokens(state, tokens)

        return state
