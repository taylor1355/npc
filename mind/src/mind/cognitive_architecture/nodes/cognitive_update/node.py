"""Cognitive update node implementation"""

from pathlib import Path
from pprint import pformat

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from mind.cognitive_architecture.nodes.base import LLMNode
from mind.cognitive_architecture.state import PipelineState
from mind.knowledge import KnowledgeBase, KnowledgeFile
from mind.logging_config import get_logger

from .models import CognitiveUpdateOutput

logger = get_logger()


class CognitiveUpdateNode(LLMNode):
    """Updates cognitive context based on memories and observations"""

    step_name = "cognitive_update"

    def __init__(self, llm: BaseChatModel):
        # Load prompt template
        prompt_path = Path(__file__).parent / "prompt.md"
        prompt = PromptTemplate.from_template(prompt_path.read_text())

        super().__init__(llm, prompt, output_model=CognitiveUpdateOutput, max_retries=2)

    async def process(self, state: PipelineState) -> PipelineState:
        """Update cognitive context with retrieved memories and observations"""

        # Format memories for prompt
        memories_text = "\n".join([str(m) for m in state.retrieved_memories])
        if not memories_text:
            memories_text = "No relevant memories found."

        # Build world knowledge from centralized knowledge files
        world_knowledge = KnowledgeBase.get([
            KnowledgeFile.NEEDS,
            KnowledgeFile.INTERACTIONS,
            KnowledgeFile.MOVEMENT,
            KnowledgeFile.ACTIVITY,
        ])

        # Generate cognitive update
        output = await self.call_llm(
            state,
            working_memory=str(state.working_memory),
            retrieved_memories=memories_text,
            observation_text=str(state.observation),
            recent_events=pformat(state.recent_events),
            world_knowledge=world_knowledge,
            format_instructions=self.get_format_instructions()
        )

        # Update state with new working memory
        state.working_memory = output.updated_working_memory

        # Add new memories to daily buffer
        state.daily_memories.extend(output.new_memories)

        # Log updated working memory
        wm = output.updated_working_memory
        logger.debug("Updated working memory:")
        logger.debug(f"  Situation: {wm.situation_assessment}")
        logger.debug(f"  Active goals: {wm.active_goals}")
        logger.debug(f"  Current plan: {wm.current_plan}")
        logger.debug(f"  Emotional state: {wm.emotional_state}")

        # Log new memories
        logger.debug(f"Storing {len(output.new_memories)} new memories")
        for mem in output.new_memories:
            logger.debug(f"  [importance={mem.importance}] {mem.content}")

        return state
