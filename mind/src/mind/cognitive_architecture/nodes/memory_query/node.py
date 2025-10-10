"""Memory query generation node"""

from langchain_core.language_models import BaseChatModel

from ..base import LLMNode
from ...state import PipelineState
from .models import MemoryQueryOutput


class MemoryQueryNode(LLMNode):
    """Generates diverse queries to search long-term memory"""

    step_name = "memory_query"
    PROMPT_VARS = {"working_memory", "observation"}

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, output_model=MemoryQueryOutput)

    async def process(self, state: PipelineState) -> PipelineState:
        """Generate memory queries from observation"""
        output, tokens = await self.call_llm(
            working_memory=str(state.working_memory),
            observation=str(state.observation)
        )

        state.memory_queries = output.queries
        self.track_tokens(state, tokens)
        return state
