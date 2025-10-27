"""Memory retrieval node implementation"""

from typing import Protocol

from ...memory.vector_db_memory import VectorDBQuery
from ...models import Memory
from ...state import PipelineState
from ..base import Node

# Default memories to retrieve per query
DEFAULT_MEMORIES_PER_QUERY = 2


class MemoryStoreProtocol(Protocol):
    """Protocol for memory storage backends"""

    async def search(self, query: VectorDBQuery) -> list[Memory]:
        """Search for memories"""
        ...


class MemoryRetrievalNode(Node):
    """Retrieves memories from storage based on queries"""

    step_name = "memory_retrieval"

    def __init__(
        self,
        memory_store: MemoryStoreProtocol,
        memories_per_query: int = DEFAULT_MEMORIES_PER_QUERY,
    ):
        self.memory_store = memory_store
        self.memories_per_query = memories_per_query

    async def process(self, state: PipelineState) -> PipelineState:
        """Retrieve memories using the queries in state"""

        # Retrieve memories for each query
        all_memories = []
        for query_text in state.memory_queries:
            query = VectorDBQuery(
                query=query_text,
                top_k=self.memories_per_query,
                current_simulation_time=state.observation.current_simulation_time,
            )
            results = await self.memory_store.search(query)
            all_memories.extend(results)

        # Deduplicate by memory ID, keeping first occurrence
        seen_ids = set()
        deduplicated_memories = []
        for memory in all_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                deduplicated_memories.append(memory)

        # Update state
        state.retrieved_memories = deduplicated_memories

        return state
