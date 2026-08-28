"""Memory retrieval node implementation"""

from typing import Protocol

from mind.constants import DEFAULT_MEMORIES_PER_QUERY
from mind.logging_config import get_logger

from ...memory import Memory
from ...memory.retrieval import RetrievalWeights
from ...memory.vector_db_memory import VectorDBQuery
from ...state import PipelineState
from ..base import Node, entity_tag

logger = get_logger()


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
        weights: RetrievalWeights | None = None,
    ):
        self.memory_store = memory_store
        self.memories_per_query = memories_per_query
        # Resolved once at construction so every query in a cycle scores on the
        # same weights, and so a per-mind override cannot be half-applied.
        self.weights = weights or RetrievalWeights()

    async def process(self, state: PipelineState) -> PipelineState:
        """Retrieve memories using the queries in state"""

        # Retrieve memories for each query
        all_memories = []
        for query_text in state.memory_queries:
            query = VectorDBQuery(
                query=query_text,
                top_k=self.memories_per_query,
                weights=self.weights,
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

        # Log retrieved memories
        tag = entity_tag(state)
        logger.debug(
            f"{tag} Retrieved {len(deduplicated_memories)} memories from {len(state.memory_queries)} queries"
        )
        # Abstention visibility. A memory with no timestamp had no recency score,
        # and one with no importance had no importance score - the scorer ranked
        # them on what was left. Both are symptoms of a write path, not of the
        # memories, so a store where the counts equal the total is a wiring bug
        # that would otherwise look exactly like correct relevance-only ranking.
        # Derived from the returned memories rather than plumbed through the
        # store, so this stays attributable to the entity (NPC-789).
        if deduplicated_memories:
            no_timestamp = sum(1 for m in deduplicated_memories if m.timestamp is None)
            no_importance = sum(1 for m in deduplicated_memories if m.importance is None)
            if no_timestamp or no_importance:
                logger.debug(
                    f"{tag} Retrieval abstentions among {len(deduplicated_memories)} memories: "
                    f"{no_timestamp} without a timestamp (recency), "
                    f"{no_importance} without an importance rating"
                )

        for i, mem in enumerate(deduplicated_memories[:3]):  # Show top 3
            content_preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
            logger.debug(f"{tag}   [{i + 1}] {content_preview}")

        return state
