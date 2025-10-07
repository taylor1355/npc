"""Memory consolidation node - processes daily memories into long-term storage"""

from ..base import Node
from ...state import PipelineState
from ...memory.vector_db_memory import VectorDBMemory


class MemoryConsolidationNode(Node):
    """Consolidates daily memories into long-term storage

    TODO: Implement sophisticated consolidation inspired by Generative Agents paper:
    - Filter/merge similar memories
    - Generate reflections/insights
    - Apply forgetting curve
    - Create higher-level abstractions

    Current implementation: Simple placeholder that adds all daily memories to long-term storage
    """

    step_name = "memory_consolidation"

    def __init__(self, memory_store: VectorDBMemory):
        self.memory_store = memory_store

    async def process(self, state: PipelineState) -> PipelineState:
        """Consolidate daily memories into long-term storage"""

        # Add all daily memories to long-term storage
        for new_memory in state.daily_memories:
            self.memory_store.add_memory(
                content=new_memory.content,
                importance=new_memory.importance,
                timestamp=state.observation_context.current_simulation_time,
                location=state.observation_context.agent_location
            )

        # Clear daily buffer
        state.daily_memories.clear()

        return state
