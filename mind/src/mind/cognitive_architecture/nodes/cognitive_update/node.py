"""Cognitive update node implementation"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from mind.cognitive_architecture.nodes.base import LLMNode
from mind.cognitive_architecture.state import PipelineState

from .models import CognitiveUpdateOutput

# Cognitive context dict keys
KEY_SITUATION = "situation_assessment"
KEY_GOALS = "current_goals"
KEY_EMOTIONAL = "emotional_state"


class CognitiveUpdateNode(LLMNode):
    """Updates cognitive context based on memories and observations"""

    step_name = "cognitive_update"

    def __init__(self, llm: BaseChatModel):
        # Load prompt template
        prompt_path = Path(__file__).parent / "prompt.md"
        prompt = PromptTemplate.from_template(prompt_path.read_text())

        super().__init__(llm, prompt, output_model=CognitiveUpdateOutput)

    async def process(self, state: PipelineState) -> PipelineState:
        """Update cognitive context with retrieved memories and observations"""

        # Format memories for prompt
        memories_text = "\n".join([str(m) for m in state.retrieved_memories])
        if not memories_text:
            memories_text = "No relevant memories found."

        # Generate cognitive update
        output = await self.call_llm(
            state,
            working_memory=str(state.working_memory),
            retrieved_memories=memories_text,
            observation_text=str(state.observation),
            format_instructions=PydanticOutputParser(pydantic_object=CognitiveUpdateOutput).get_format_instructions()
        )

        # Update state
        state.working_memory = output.updated_working_memory
        state.cognitive_context = {
            KEY_SITUATION: output.situation_assessment,
            KEY_GOALS: output.current_goals,
            KEY_EMOTIONAL: output.emotional_state,
        }

        # Add new memories to daily buffer
        state.daily_memories.extend(output.new_memories)

        return state
