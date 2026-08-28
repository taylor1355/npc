"""Memory consolidation node - processes daily memories into long-term storage"""

from ...memory.vector_db_memory import VectorDBMemory
from ...state import PipelineState
from ..base import Node


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

    def __init__(self, memory_store: VectorDBMemory, write_timestamp: int | None):
        """
        Args:
            memory_store: Where consolidated memories land.
            write_timestamp: Elapsed game minutes to stamp these memories with,
                or None when the caller genuinely does not know.

                Required, with no default, deliberately. This node runs outside
                the graph, driven by a caller that assembles a state object for
                it, and it used to read the stamp off that state's observation -
                which the only production caller fabricated with
                current_simulation_time=0. Every lived memory was therefore
                written at the epoch and decayed from it, while config-seeded
                memories carrying no timestamp at all scored *perfect* recency,
                so hardcoded backstory permanently outranked lived experience.
                Making the stamp an explicit argument means no caller can supply
                one by accident, and None travels through as an honest
                abstention instead of as a fake zero.
        """
        self.memory_store = memory_store
        self.write_timestamp = write_timestamp

    async def process(self, state: PipelineState) -> PipelineState:
        """Consolidate daily memories into long-term storage"""

        # Add all daily memories to long-term storage
        for new_memory in state.daily_memories:
            # Extract location from status observation if available
            location = None
            if state.observation.status:
                location = state.observation.status.position

            self.memory_store.add_memory(
                content=new_memory.content,
                importance=new_memory.importance,
                timestamp=self.write_timestamp,
                location=location,
            )

        # Clear daily buffer
        state.daily_memories.clear()

        return state
