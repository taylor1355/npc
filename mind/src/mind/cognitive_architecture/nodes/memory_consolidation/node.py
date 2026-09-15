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

    What it does NOT do, deliberately: read circumstances off the observation it
    is handed. Where and when a memory happened are properties of its FORMATION,
    and this node runs a whole day later over a whole batch - so it copies each
    memory's own stamp through rather than measuring anything itself (NPC-1476).
    The one exception is `write_timestamp`, which is genuinely a property of the
    write and is an explicit constructor argument for the reasons below.
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

        # Add all daily memories to long-term storage.
        #
        # Place and position come from each memory's OWN formation stamp, never
        # from `state.observation` (NPC-1476). This node runs once per day over a
        # whole batch, so a single read out here stamped every memory of the day
        # with wherever the NPC happened to be standing when consolidation fired
        # - the batch inherited one cell. The stamp is taken at formation
        # instead, in ReflectionNode, where the observation is live and actually
        # describes the circumstances of that memory.
        #
        # None flows through untouched: unstamped is the honest reading of a
        # memory formed outside any zone, or of one written before this existed,
        # and the retrieval scorer abstains on it rather than scoring it.
        for formed_memory in state.daily_memories:
            self.memory_store.add_memory(
                content=formed_memory.content,
                importance=formed_memory.importance,
                timestamp=self.write_timestamp,
                location=formed_memory.formed_at_position,
                zone_id=formed_memory.formed_in_zone_id,
            )

        # Clear daily buffer
        state.daily_memories.clear()

        return state
