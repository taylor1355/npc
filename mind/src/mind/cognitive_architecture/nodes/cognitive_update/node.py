"""Cognitive update node implementation"""

from pathlib import Path
from pprint import pformat

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from mind.cognitive_architecture.nodes.base import LLMNode, entity_tag
from mind.cognitive_architecture.nodes.formatting import (
    format_interaction_status as _format_interaction_status,
)
from mind.cognitive_architecture.nodes.formatting import (
    format_personality,
)
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

        # Format personality for the prompt (shared helper keeps the rendered
        # representation identical to action_selection across the pipeline)
        personality_text, dims_text = format_personality(
            state.personality_traits, state.personality_dimensions
        )

        # Build world knowledge from centralized knowledge files
        world_knowledge = KnowledgeBase.get([
            KnowledgeFile.NEEDS,
            KnowledgeFile.INTERACTIONS,
            KnowledgeFile.MOVEMENT,
            KnowledgeFile.ACTIVITY,
        ])

        # Ground the "am I interacting?" belief in the fresh observation so a
        # stale working-memory belief ("I'm in a conversation") gets corrected
        # this cycle rather than persisting after teardown (NPC-688).
        interaction_status = _format_interaction_status(state.observation)

        # Generate cognitive update
        output = await self.call_llm(
            state,
            working_memory=str(state.working_memory),
            personality_traits=personality_text,
            personality_dimensions=dims_text,
            retrieved_memories=memories_text,
            observation_text=str(state.observation),
            interaction_status=interaction_status,
            recent_events=pformat(state.recent_events),
            world_knowledge=world_knowledge,
            format_instructions=self.get_format_instructions()
        )

        # Update state with new working memory
        state.working_memory = output.updated_working_memory

        # Add new memories to daily buffer
        state.daily_memories.extend(output.new_memories)

        # Log updated working memory as a single record so the simulation's
        # Events tab shows one entry per thought instead of one per field
        tag = entity_tag(state)
        wm = output.updated_working_memory
        logger.debug(
            f"{tag} Updated working memory:\n"
            f"  Situation: {wm.situation_assessment}\n"
            f"  Active goals: {wm.active_goals}\n"
            f"  Current plan: {wm.current_plan}\n"
            f"  Emotional state: {wm.emotional_state}"
        )

        # Log new memories as a single record for the same reason
        memory_lines = "".join(
            f"\n  [importance={mem.importance}] {mem.content}"
            for mem in output.new_memories
        )
        logger.debug(f"{tag} Storing {len(output.new_memories)} new memories{memory_lines}")

        return state
