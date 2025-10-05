"""Cognitive update node implementation"""

from langchain_core.language_models import BaseChatModel

from ..base import LLMNode
from ...state import PipelineState
from .models import CognitiveUpdateOutput

# Cognitive context dict keys
KEY_SITUATION = "situation_assessment"
KEY_GOALS = "current_goals"
KEY_EMOTIONAL = "emotional_state"


class CognitiveUpdateNode(LLMNode):
    """Updates cognitive context based on memories and observations"""

    step_name = "cognitive_update"
    PROMPT_VARS = {"working_memory", "retrieved_memories", "observation_text"}

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, output_model=CognitiveUpdateOutput)

    async def process(self, state: PipelineState) -> PipelineState:
        """Update cognitive context with retrieved memories and observations"""

        # Format memories for prompt
        memories_text = "\n".join([f"- {m.content}" for m in state.retrieved_memories])
        if not memories_text:
            memories_text = "No relevant memories found."

        # Generate cognitive update
        output, tokens = await self.call_llm(
            working_memory=state.working_memory,
            retrieved_memories=memories_text,
            observation_text=state.observation_context.observation_text
        )

        # Update state
        state.working_memory = output.updated_working_memory
        state.cognitive_context = {
            KEY_SITUATION: output.situation_assessment,
            KEY_GOALS: output.current_goals,
            KEY_EMOTIONAL: output.emotional_state
        }
        self.track_tokens(state, tokens)

        return state