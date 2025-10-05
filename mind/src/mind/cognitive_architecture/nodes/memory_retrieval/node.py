"""Memory retrieval node implementation"""

from typing import Protocol

from ..base import Node
from ...state import PipelineState
from ...models import Memory

# Default memories to retrieve per query
DEFAULT_MEMORIES_PER_QUERY = 2


class MemoryStoreProtocol(Protocol):
    """Protocol for memory storage backends"""

    async def search(self, query: str, top_k: int = 5) -> list[Memory]:
        """Search for memories"""
        ...


class MemoryRetrievalNode(Node):
    """Retrieves memories from storage based on queries"""

    step_name = "memory_retrieval"

    def __init__(self, memory_store: MemoryStoreProtocol, memories_per_query: int = DEFAULT_MEMORIES_PER_QUERY):
        self.memory_store = memory_store
        self.memories_per_query = memories_per_query

    async def process(self, state: PipelineState) -> PipelineState:
        """Retrieve memories using the queries in state"""

        memories = []

        # Retrieve memories for each query
        for query in state.memory_queries:
            results = await self.memory_store.search(query, top_k=self.memories_per_query)
            memories.extend(results)

        # Update state
        state.retrieved_memories = memories

        return state