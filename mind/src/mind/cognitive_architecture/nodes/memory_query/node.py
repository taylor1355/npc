"""Memory query generation node"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from mind.cognitive_architecture.nodes.base import LLMNode
from mind.cognitive_architecture.state import PipelineState

from .models import MemoryQueryOutput


class MemoryQueryNode(LLMNode):
    """Generates diverse queries to search long-term memory"""

    step_name = "memory_query"

    def __init__(self, llm: BaseChatModel):
        # Load prompt template
        prompt_path = Path(__file__).parent / "prompt.md"
        prompt = PromptTemplate.from_template(prompt_path.read_text())

        super().__init__(llm, prompt, output_model=MemoryQueryOutput)

    async def process(self, state: PipelineState) -> PipelineState:
        """Generate memory queries from observation"""
        output = await self.call_llm(
            state,
            working_memory=str(state.working_memory),
            observation=str(state.observation),
            format_instructions=PydanticOutputParser(pydantic_object=MemoryQueryOutput).get_format_instructions()
        )

        state.memory_queries = output.queries
        return state
